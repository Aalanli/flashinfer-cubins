"""Single-kernel ELF strip for CUDA cubins (compile stage only).

``strip_cubin(image, keep)`` deletes every kernel of a cubin except ``keep``:
the dead kernels' sections, symbols, relocations, ``.nv.info``/callgraph
entries, frame descriptors and line sequences. Sections and symbols are
renumbered consistently and every other byte is kept, including the kept
kernel's code, constant bank, metadata and Mercury capsule. Only the capsule's
leading word changes, since it holds the index of its SASS ``.text`` section.
``check_strip`` verifies these invariants. The strip was introduced for
``impls/moe`` and is shared by every package that splits a precompiled
multi-kernel cubin (trtllm-gen ``GetSmemSize`` helpers, FlashInfer JIT-cache
modules).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

SHT_NULL, SHT_PROGBITS, SHT_SYMTAB, SHT_RELA, SHT_NOBITS, SHT_REL = 0, 1, 2, 4, 8, 9
SHT_NOTE = 7
SHT_CUDA_INFO = 0x70000000
SHT_CUDA_CALLGRAPH = 0x70000001
SHT_CUDA_CAPMERC_TEXT = 0x70000016
SHT_CUDA_MERC_RELA = 0x70000082
SHT_CUDA_MERC_INFO = 0x70000083
SHT_CUDA_MERC_SYMTAB = 0x70000085
SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR, SHF_INFO_LINK = 0x1, 0x2, 0x4, 0x40
PF_W = 0x2
SHN_LORESERVE = 0xFF00
EIFMT_SVAL = 4
# .nv.info attributes holding symbol indices: attribute -> u32 words
# (None: every word).
EIATTR_SYMBOL_WORDS: dict[int, tuple[int, ...] | None] = {
    0x0A: (0,),  # EIATTR_PARAM_CBANK (constant-bank section symbol)
    0x0F: None,  # EIATTR_EXTERNS
    0x11: (0,),  # EIATTR_FRAME_SIZE
    0x12: (0,),  # EIATTR_MIN_STACK_SIZE
    0x23: (0,),  # EIATTR_MAX_STACK_SIZE
    0x2F: (0,),  # EIATTR_REGCOUNT
}
# Word-valued attributes checked to hold neither symbol nor section indices
# (instruction offsets, sizes, masks, versions). Anything else is refused.
EIATTR_PLAIN = frozenset(
    {
        0x04,  # CTAIDZ_USED
        0x05,  # MAX_THREADS
        0x0C,  # CBANK_PARAM_OFFSETS
        0x10,  # REQNTID
        0x17,  # KPARAM_INFO
        0x19,  # CBANK_PARAM_SIZE
        0x1B,  # MAXREG_COUNT
        0x1C,  # EXIT_INSTR_OFFSETS
        0x1D,  # S2RCTAID_INSTR_OFFSETS
        0x1E,  # CRS_STACK_SIZE
        0x28,  # COOP_GROUP_INSTR_OFFSETS
        0x29,  # COOP_GROUP_MASK_REGIDS
        0x31,  # INT_WARP_WIDE_INSTR_OFFSETS
        0x34,  # INDIRECT_BRANCH_TARGETS (instruction offsets)
        0x35,  # SW2861232_WAR
        0x36,  # SW_WAR
        0x37,  # CUDA_API_VERSION
        0x38,  # NUM_MBARRIERS
        0x39,  # MBARRIER_INSTR_OFFSETS
        0x3D,  # CTA_PER_CLUSTER
        0x3E,  # EXPLICIT_CLUSTER
        0x3F,  # MAX_CLUSTER_RANK
        0x41,  # RESERVED_SMEM_USED
        0x44,  # UNUSED_LOAD_BYTE_OFFSET
        0x45,  # KPARAM_INFO_V2
        0x46,  # SYSCALL_OFFSETS
        0x4A,  # VRC_CTA_INIT_COUNT
        0x4C,  # NUM_BARRIERS
        0x4F,  # AT_ENTRY_FRAGMENTS
        0x50,  # SPARSE_MMA_MASK
        0x51,  # TCGEN05_1CTA_USED
        0x54,  # REG_RECONFIG
        0x55,  # (instruction offset list)
        0x5A,  # (Mercury digest)
        0x5F,  # MERCURY_ISA_VERSION
        0x66,  # LANGUAGE
    }
)


@dataclass
class _Section:
    index: int
    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    align: int
    entsize: int
    data: bytes = b""

    @property
    def nobits(self) -> bool:
        return self.type == SHT_NOBITS


@dataclass
class _Elf:
    header: bytes
    sections: list[_Section]
    segments: list[list[int]]
    phoff: int
    shoff: int
    shstrndx: int
    raw: bytes = field(repr=False, default=b"")


def _parse_elf(image: bytes) -> _Elf:
    if image[:4] != b"\x7fELF" or image[4] != 2 or image[5] != 1:
        raise ValueError("expected a little-endian ELF64 cubin")
    phoff, shoff = struct.unpack_from("<QQ", image, 0x20)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from(
        "<HHHHH", image, 0x36
    )
    if shentsize != 64 or (phnum and phentsize != 56):
        raise ValueError("unexpected ELF header entry sizes")
    raw = [
        struct.unpack_from("<IIQQQQIIQQ", image, shoff + i * 64) for i in range(shnum)
    ]
    names = image[raw[shstrndx][4] : raw[shstrndx][4] + raw[shstrndx][5]]
    sections = []
    for i, (
        name,
        kind,
        flags,
        addr,
        off,
        size,
        link,
        info,
        align,
        entsize,
    ) in enumerate(raw):
        data = b"" if kind in (SHT_NOBITS, SHT_NULL) else image[off : off + size]
        if kind not in (SHT_NOBITS, SHT_NULL) and len(data) != size:
            raise ValueError(f"section {i} is truncated")
        label = names[name : names.index(b"\0", name)].decode()
        sections.append(
            _Section(
                i, label, kind, flags, addr, off, size, link, info, align, entsize, data
            )
        )
    segments = [
        list(struct.unpack_from("<IIQQQQQQ", image, phoff + i * 56))
        for i in range(phnum)
    ]
    return _Elf(image[:64], sections, segments, phoff, shoff, shstrndx, image)


def _symbols(section: _Section) -> list[tuple[int, int, int, int, int, int]]:
    return [
        struct.unpack_from("<IBBHQQ", section.data, i * 24)
        for i in range(section.size // 24)
    ]


def _is_symtab(section: _Section) -> bool:
    return section.type in (SHT_SYMTAB, SHT_CUDA_MERC_SYMTAB)


def _is_text(section: _Section) -> bool:
    return section.type == SHT_PROGBITS and bool(section.flags & SHF_EXECINSTR)


def cubin_kernels(image: bytes) -> list[str]:
    """Kernel names (``.text.<name>`` sections) of a cubin."""
    return [s.name[len(".text.") :] for s in _parse_elf(image).sections if _is_text(s)]


def section_bytes(image: bytes, name: str) -> bytes:
    """Contents of the section called ``name`` (KeyError if absent)."""
    for s in _parse_elf(image).sections:
        if s.name == name:
            return s.data
    raise KeyError(name)


def _frame_entries(data: bytes) -> list[tuple[int, int, bool, int, int]]:
    """(start, end, is_cie, id offset, id/CIE pointer) of each .debug_frame entry."""
    entries, pos = [], 0
    while pos < len(data):
        length = struct.unpack_from("<I", data, pos)[0]
        if length == 0xFFFFFFFF:
            length = struct.unpack_from("<Q", data, pos + 4)[0]
            id_offset, id_size, cie_id = pos + 12, 8, 0xFFFFFFFFFFFFFFFF
        elif length >= 0xFFFFFFF0:
            raise ValueError("reserved DWARF unit length in .debug_frame")
        else:
            id_offset, id_size, cie_id = pos + 4, 4, 0xFFFFFFFF
        end = id_offset + length
        if length == 0 or end > len(data):
            raise ValueError("malformed .debug_frame entry")
        value = int.from_bytes(data[id_offset : id_offset + id_size], "little")
        entries.append((pos, end, value == cie_id, id_offset, value))
        pos = end
    return entries


def _uleb(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, pos


def _line_sequences(data: bytes) -> list[tuple[int, int, int]]:
    """(unit start, sequence start, sequence end) of every DWARF line sequence.

    Supports the 32-bit DWARF v2-v4 line programs nvcc emits; anything else
    raises instead of being guessed.
    """
    out, unit = [], 0
    while unit < len(data):
        length = struct.unpack_from("<I", data, unit)[0]
        if length >= 0xFFFFFFF0:
            raise ValueError("64-bit DWARF line programs are not supported")
        end = unit + 4 + length
        version = struct.unpack_from("<H", data, unit + 4)[0]
        if not 2 <= version <= 4:
            raise ValueError(f"DWARF line program version {version} is not supported")
        header_length = struct.unpack_from("<I", data, unit + 6)[0]
        base = unit + 10 + (1 if version >= 4 else 0)  # v4: maximum_ops_per_insn
        opcode_base = data[base + 4]
        lengths = data[base + 5 : base + 5 + opcode_base - 1]
        pos = start = unit + 10 + header_length
        while pos < end:
            op = data[pos]
            pos += 1
            if op == 0:  # extended opcode
                size, pos = _uleb(data, pos)
                sub = data[pos]
                pos += size
                if sub == 1:  # DW_LNE_end_sequence
                    out.append((unit, start, pos))
                    start = pos
            elif op < opcode_base:
                for _ in range(lengths[op - 1]):
                    _, pos = _uleb(data, pos)
        if pos != end or start != end:
            raise ValueError("line program does not end with a complete sequence")
        unit = end
    return out


def _nobits_size(sections: list[_Section], members: list[int], size: int = 0) -> int:
    """Memory size of a segment: ``size`` file bytes, then NOBITS ``members``."""
    for i in sorted(members):
        align = max(1, sections[i].align)
        size = -(-size // align) * align + sections[i].size
    return size


def _map_offset(offset: int, spans: list[tuple[int, int, int]]) -> int | None:
    """New offset of ``offset`` under kept spans (old start, old end, new start)."""
    return next((offset - a + b for a, e, b in spans if a <= offset < e), None)


def _info_is_section(s: _Section) -> bool:
    """sh_info names a section: SHF_INFO_LINK, or a relocation section's
    target (the sm_100a trtllm-gen FMHA cubins omit the flag on those)."""
    return bool(s.flags & SHF_INFO_LINK) or (
        s.type in (SHT_RELA, SHT_REL, SHT_CUDA_MERC_RELA) and s.info != 0
    )


def strip_cubin(image: bytes, keep: str | None) -> bytes:
    """Remove every kernel of ``image`` except ``keep`` (``None``: rebuild only).

    Deleted per dead kernel: its ``.text`` and Mercury ``.nv.capmerc.text``
    sections, every section info-linked to them (``.nv.info.<k>``,
    ``.nv.shared.<k>``, ``.nv.constant0.<k>``, ``.rela.text.<k>``,
    ``.nv.merc.*.<k>``, ...), the symbols they define, their entries in the
    global ``.nv.info``/``.nv.merc.nv.info``/``.nv.callgraph`` tables, their
    ``.debug_frame`` FDEs (and CIEs left unused), their DWARF line sequences
    and the relocations of those. Sections, symbols, relocation symbol
    indices, ``.nv.info`` symbol references, ``sh_link``/``sh_info``, segment
    membership and the Mercury capsules' leading SASS-section index are
    renumbered; every other byte is copied unchanged (string tables included).
    Unknown attributes or layouts raise instead of being guessed.
    """
    elf = _parse_elf(image)
    sections = elf.sections
    names = [s.name[len(".text.") :] for s in sections if _is_text(s)]
    if keep is not None and keep not in names:
        raise ValueError(f"{keep!r} is not a kernel of this cubin: {names}")
    dead = [k for k in names if keep is not None and k != keep]
    by_name = {s.name: s.index for s in sections}
    if len(by_name) != len(sections):
        raise ValueError("duplicate section names")

    removed: set[int] = set()
    for k in dead:
        removed.add(by_name[".text." + k])
        if ".nv.capmerc.text." + k in by_name:
            removed.add(by_name[".nv.capmerc.text." + k])
    while True:
        more = {
            s.index
            for s in sections
            if s.index not in removed and _info_is_section(s) and s.info in removed
        }
        if not more:
            break
        removed |= more
    for s in sections:
        if s.index in removed:
            continue
        if any(s.name.endswith("." + k) for k in dead):
            raise ValueError(f"{s.name} names a removed kernel but is not linked to it")
        if s.link in removed:
            raise ValueError(f"{s.name} links to a removed section")
    if elf.shstrndx in removed:
        raise ValueError("cannot remove the section name table")
    section_map = {
        s.index: n for n, s in enumerate(x for x in sections if x.index not in removed)
    }

    # Symbols defined in removed sections disappear; the rest are renumbered.
    symbol_maps: dict[int, dict[int, int]] = {}
    removed_symbols: dict[int, set[int]] = {}
    new_data: dict[int, bytes] = {}
    new_info: dict[int, int] = {}
    for s in sections:
        if not _is_symtab(s) or s.index in removed:
            continue
        mapping: dict[int, int] = {}
        gone: set[int] = set()
        out = bytearray()
        first_global = 0
        for i, (name, info, other, shndx, value, size) in enumerate(_symbols(s)):
            if 0 < shndx < SHN_LORESERVE and shndx in removed:
                gone.add(i)
                continue
            first_global += i < s.info
            mapping[i] = len(mapping)
            if 0 < shndx < SHN_LORESERVE:
                shndx = section_map[shndx]
            out += struct.pack("<IBBHQQ", name, info, other, shndx, value, size)
        symbol_maps[s.index], removed_symbols[s.index] = mapping, gone
        new_data[s.index], new_info[s.index] = bytes(out), first_global

    # Some cubins (the trtllm-gen sm_100a FMHA ones) leave sh_link of the
    # global .nv.info and .nv.callgraph at 0; they index the main .symtab.
    symtab = next(s.index for s in sections if s.name == ".symtab")

    def symbol(table: int, index: int, what: str) -> int:
        if index in removed_symbols[table]:
            raise ValueError(f"{what} references removed symbol {index}")
        return symbol_maps[table][index]

    relocations = {
        s.index: s
        for s in sections
        if s.type in (SHT_RELA, SHT_REL, SHT_CUDA_MERC_RELA) and s.index not in removed
    }

    def dead_offsets(target: int) -> set[int]:
        """Offsets in ``target`` relocated against removed symbols."""
        offsets = set()
        for r in relocations.values():
            if r.info != target:
                continue
            entsize = 16 if r.type == SHT_REL else 24
            for j in range(r.size // entsize):
                offset, info = struct.unpack_from("<QQ", r.data, j * entsize)
                if (info >> 32) in removed_symbols[r.link]:
                    offsets.add(offset)
        return offsets

    spans_of: dict[int, list[tuple[int, int, int]]] = {}
    for target in sorted({r.info for r in relocations.values()}):
        t = sections[target]
        doomed = dead_offsets(target)
        if t.name.endswith("debug_frame"):
            entries = _frame_entries(t.data)
            drop = {
                n
                for n, (a, e, is_cie, _, _) in enumerate(entries)
                if not is_cie and any(a <= o < e for o in doomed)
            }
            used = {
                v
                for n, (_, _, c, _, v) in enumerate(entries)
                if not c and n not in drop
            }
            if not used <= {a for a, _, c, _, _ in entries if c}:
                raise ValueError("FDE CIE pointer is not a CIE offset")
            drop |= {
                n for n, (a, _, c, _, _) in enumerate(entries) if c and a not in used
            }
            spans: list[tuple[int, int, int]] = []
            out = bytearray()
            for n, (a, e, _, _, _) in enumerate(entries):
                if n not in drop:
                    spans.append((a, e, len(out)))
                    out += t.data[a:e]
            for n, (a, _, is_cie, id_offset, value) in enumerate(entries):
                if n in drop or is_cie:
                    continue
                pointer = _map_offset(value, spans)
                new_start = _map_offset(a, spans)
                if pointer is None or new_start is None:
                    raise ValueError("FDE references a removed CIE")
                size = 8 if id_offset - a == 12 else 4
                at = new_start + id_offset - a
                out[at : at + size] = pointer.to_bytes(size, "little")
        elif "debug_line" in t.name:
            sequences = _line_sequences(t.data)
            dropped = [
                (a, c) for _, a, c in sequences if any(a <= o < c for o in doomed)
            ]
            if len(dropped) != len(doomed):
                raise ValueError(f"{t.name}: dead relocation outside one sequence")
            spans, out, pos = [], bytearray(), 0
            for a, c in dropped:
                spans.append((pos, a, len(out)))
                out += t.data[pos:a]
                pos = c
            spans.append((pos, len(t.data), len(out)))
            out += t.data[pos:]
            for unit in sorted({u for u, _, _ in sequences}):
                old_end = unit + 4 + struct.unpack_from("<I", t.data, unit)[0]
                cut = sum(c - a for a, c in dropped if unit <= a < old_end)
                new_unit = _map_offset(unit, spans)
                assert new_unit is not None
                struct.pack_into("<I", out, new_unit, old_end - unit - 4 - cut)
        else:
            if doomed:
                raise ValueError(f"{t.name} is relocated against a removed symbol")
            continue
        new_data[target] = bytes(out)
        spans_of[target] = spans

    for r in relocations.values():
        entsize = 16 if r.type == SHT_REL else 24
        out = bytearray()
        spans_or_none = spans_of.get(r.info)
        for j in range(r.size // entsize):
            chunk = bytearray(r.data[j * entsize : (j + 1) * entsize])
            offset, info = struct.unpack_from("<QQ", chunk, 0)
            sym = info >> 32
            if spans_or_none is not None:
                moved = _map_offset(offset, spans_or_none)
                if moved is None:
                    if sym not in removed_symbols[r.link]:
                        raise ValueError(f"{r.name}: live relocation in removed entry")
                    continue
                offset = moved
            sym = symbol(r.link, sym, r.name)
            struct.pack_into("<QQ", chunk, 0, offset, (sym << 32) | (info & 0xFFFFFFFF))
            out += chunk
        new_data[r.index] = bytes(out)

    for s in sections:
        if s.index in removed:
            continue
        table = s.link or symtab
        if s.type in (SHT_CUDA_INFO, SHT_CUDA_MERC_INFO):
            per_kernel = bool(s.flags & SHF_INFO_LINK)
            out, pos, data = bytearray(), 0, s.data
            while pos < len(data):
                fmt, attr = data[pos], data[pos + 1]
                if fmt != EIFMT_SVAL:  # 1/2-byte values: no indices
                    out += data[pos : pos + 4]
                    pos += 4
                    continue
                length = struct.unpack_from("<H", data, pos + 2)[0]
                head = data[pos : pos + 4]
                payload = bytearray(data[pos + 4 : pos + 4 + length])
                pos += 4 + length
                if attr in EIATTR_SYMBOL_WORDS:
                    words = EIATTR_SYMBOL_WORDS[attr] or tuple(range(length // 4))
                    refs = [struct.unpack_from("<I", payload, 4 * w)[0] for w in words]
                    if any(v in removed_symbols[table] for v in refs):
                        if per_kernel:
                            raise ValueError(f"{s.name} references a removed symbol")
                        continue  # a dead kernel's entry in the global table
                    for w, v in zip(words, refs):
                        struct.pack_into("<I", payload, 4 * w, symbol(table, v, s.name))
                elif attr not in EIATTR_PLAIN:
                    raise ValueError(f"{s.name}: unknown attribute {attr:#x}")
                out += head + payload
            new_data[s.index] = bytes(out)
        elif s.type == SHT_CUDA_CALLGRAPH:
            out = bytearray()
            for j in range(s.size // 8):
                pair = struct.unpack_from("<II", s.data, 8 * j)
                refs = [v for v in pair if v < 0xFFFFFFF0]  # others are markers
                if any(v in removed_symbols[table] for v in refs):
                    continue
                out += struct.pack(
                    "<II",
                    *(symbol(table, v, s.name) if v < 0xFFFFFFF0 else v for v in pair),
                )
            new_data[s.index] = bytes(out)
        elif s.type == SHT_CUDA_CAPMERC_TEXT:
            # The Mercury capsule's first word is its SASS .text section index.
            text = by_name[".text." + s.name[len(".nv.capmerc.text.") :]]
            if struct.unpack_from("<I", s.data)[0] != text:
                raise ValueError(f"{s.name}: unexpected capsule header")
            new_data[s.index] = struct.pack("<I", section_map[text]) + s.data[4:]

    headers: dict[int, tuple[int, int]] = {}
    for s in sections:
        if s.index in removed or s.type == SHT_NULL:
            continue
        link = section_map[s.link] if s.link else 0
        info = s.info
        if _is_symtab(s):
            info = new_info[s.index]
        elif _is_text(s) or s.type == SHT_CUDA_CAPMERC_TEXT:
            # Low 24 bits: the function symbol; high byte (sm_8x): registers.
            info = (s.info & ~0xFFFFFF) | symbol(s.link, s.info & 0xFFFFFF, s.name)
        elif _info_is_section(s):
            info = section_map[s.info]
        elif s.type == SHT_NOTE and s.name == ".note.nv.cuver" and s.info:
            # The sm_100a trtllm-gen FMHA cubins name a section here without
            # SHF_INFO_LINK (the sm_100f ones set the flag).
            if s.info in removed:
                raise ValueError(f"{s.name} references a removed section")
            info = section_map[s.info]
        elif s.info:
            raise ValueError(f"{s.name}: unknown sh_info {s.info:#x}")
        headers[s.index] = (link, info)

    # Segments: the sections each one covers (file part + writable NOBITS tail).
    phnum_old = len(elf.segments)
    segments: list[tuple[str, list[int], list[int]]] = []
    for seg in elf.segments:
        _, flags, off, _, _, filesz, memsz, _ = seg
        if off == elf.phoff and filesz == phnum_old * 56:
            segments.append(("phdr", seg, []))
            continue
        members = [
            s.index
            for s in sections
            if s.flags & SHF_ALLOC
            and not s.nobits
            and s.size
            and off <= s.offset
            and s.offset + s.size <= off + filesz
        ]
        if members and (
            min(sections[i].offset for i in members),
            max(sections[i].offset + sections[i].size for i in members),
        ) != (off, off + filesz):
            raise ValueError("segment does not exactly cover its sections")
        tail = [
            s.index
            for s in sections
            if s.flags & SHF_ALLOC
            and s.nobits
            and s.offset == off + filesz
            and bool(s.flags & SHF_WRITE) == bool(flags & PF_W)
        ]
        if _nobits_size(sections, tail, filesz) != memsz:
            raise ValueError("segment memory size does not match its sections")
        if not members and not tail:
            raise ValueError("segment without sections")
        segments.append(("load", seg, members + tail))
    kept_segments = [
        (kind, seg, [i for i in members if i not in removed])
        for kind, seg, members in segments
        if kind == "phdr" or any(i not in removed for i in members)
    ]
    shnum, phnum = len(section_map), len(kept_segments)

    def size_of(index: int) -> int:
        s = sections[index]
        return s.size if s.nobits else len(new_data.get(index, s.data))

    # File layout in original order, keeping original gaps between kept
    # neighbours and alignment; aliased sections (same bytes) stay aliased.
    items = []
    for s in sections:
        if s.type != SHT_NULL:
            old = 0 if s.nobits else s.size
            items.append((s.offset, old, ("s", s.index), max(1, s.align)))
    items.append((elf.shoff, len(sections) * 64, ("sh", 0), 8))
    if phnum_old:
        items.append((elf.phoff, phnum_old * 56, ("ph", 0), 8))
    items.sort(key=lambda t: (t[0], t[1] > 0, t[2][1]))
    placed: dict[tuple[str, int], int] = {}
    alias: dict[tuple[int, int], int] = {}
    out = bytearray(64)
    cursor, old_end, prev_kept = 64, 64, True
    for old_off, old_size, key, align in items:
        if key[0] == "s" and key[1] in removed:
            if old_size:
                old_end, prev_kept = max(old_end, old_off + old_size), False
            continue
        if old_size and (old_off, old_size) in alias:
            placed[key] = alias[(old_off, old_size)]
            continue
        if not old_size:  # NOBITS / empty sections: a position, no bytes
            # The sm_100a trtllm-gen FMHA cubins give NOBITS sections offset 0.
            placed[key] = -(-cursor // align) * align if old_off else 0
            continue
        if old_off < old_end:
            raise ValueError("overlapping sections")
        gap = old_off - old_end if prev_kept else 0
        new_off = -(-(cursor + gap) // align) * align
        if gap and new_off == cursor + gap:
            out += elf.raw[old_end:old_off]  # keep the original gap bytes
        else:
            out += bytes(new_off - cursor)
        placed[key] = new_off
        if key[0] == "s":
            size = size_of(key[1])
            out += new_data.get(key[1], sections[key[1]].data)
            alias[(old_off, old_size)] = new_off
        else:
            size = (shnum * 64) if key[0] == "sh" else (phnum * 56)
            out += bytes(size)
        cursor, old_end, prev_kept = new_off + size, old_off + old_size, True

    shoff, phoff = placed[("sh", 0)], placed.get(("ph", 0), 0)
    for s in sections:
        if s.index in removed:
            continue
        name_off = struct.unpack_from("<I", elf.raw, elf.shoff + s.index * 64)[0]
        if s.type == SHT_NULL:
            entry = (name_off, 0, s.flags, s.addr, 0, 0, 0, 0, s.align, s.entsize)
        else:
            link, info = headers[s.index]
            entry = (
                name_off, s.type, s.flags, s.addr, placed[("s", s.index)],
                size_of(s.index), link, info, s.align, s.entsize,
            )  # fmt: skip
        struct.pack_into("<IIQQQQIIQQ", out, shoff + section_map[s.index] * 64, *entry)
    for n, (kind, seg, members) in enumerate(kept_segments):
        ptype, flags, off, vaddr, paddr, filesz, memsz, align = seg
        if kind == "phdr":
            off, filesz, memsz = phoff, phnum * 56, phnum * 56
        else:
            files = [i for i in members if not sections[i].nobits]
            tail = [i for i in members if sections[i].nobits]
            if files:
                off = min(placed[("s", i)] for i in files)
                filesz = max(placed[("s", i)] + size_of(i) for i in files) - off
            else:
                off, filesz = placed[("s", tail[0])], 0
            memsz = _nobits_size(sections, tail, filesz)
        struct.pack_into(
            "<IIQQQQQQ", out, phoff + 56 * n,
            ptype, flags, off, vaddr, paddr, filesz, memsz, align,
        )  # fmt: skip
    header = bytearray(elf.header)
    struct.pack_into("<QQ", header, 0x20, phoff, shoff)
    struct.pack_into("<H", header, 0x38, phnum)
    struct.pack_into("<HH", header, 0x3C, shnum, section_map[elf.shstrndx])
    out[:64] = header
    return bytes(out)


def check_strip(original: bytes, stripped: bytes, keep: str) -> None:
    """The invariants ``compile`` requires of every stripped cubin."""
    if strip_cubin(original, None) != original:
        raise ValueError("identity rebuild does not reproduce the cubin")
    if cubin_kernels(stripped) != [keep]:
        raise ValueError(f"stripped cubin holds {cubin_kernels(stripped)}")
    for prefix in (".text.", ".nv.constant0.", ".nv.merc.nv.info."):
        try:
            before = section_bytes(original, prefix + keep)
        except KeyError:
            continue
        if section_bytes(stripped, prefix + keep) != before:
            raise ValueError(f"{prefix}{keep} changed")
    try:
        capsule = section_bytes(original, ".nv.capmerc.text." + keep)
    except KeyError:
        capsule = None
    if capsule is not None and (
        section_bytes(stripped, ".nv.capmerc.text." + keep)[4:] != capsule[4:]
    ):
        raise ValueError("Mercury capsule changed beyond its section index")
