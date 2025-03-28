from __future__ import annotations
import struct
from abc import ABC, abstractmethod

class Stream:
    def __init__(self, file:str):
        if not isinstance(file, str):
            raise ValueError(f"Expected a string, but it was {type(file)}.")
        
        self._file = open(file, 'rb')
    
    @property
    def position(self):
        return self._file.tell()
    
    def jump(self, pos:int):
        return self._file.seek(pos, 0)
    
    def close(self):
        self._file.close()
    
    def read(self, bytes=1):
        res = self._file.read(bytes)
        return res
    
    def read_byte(self):
        bytes = self.read(1)
        return struct.unpack("<B", bytes)[0]
    
    def read_short(self):
        bytes = self.read(2)
        return struct.unpack("<H", bytes)[0]
    
    def read_int(self):
        bytes = self.read(4)
        return struct.unpack("<I", bytes)[0]
    
    def read_long(self):
        bytes = self.read(6)
        return struct.unpack("<Q", bytes)[0]
    
    def read_str(self, n=1):
        bytes = self.read(bytes)
        return bytes.decode('iso-8859-1')

class Header:
    def __init__(self, stream:Stream):
        self.crc = stream.read_short()
        self.id = stream.read_short()
        self.length = stream.read_int()
        self.version = stream.read_byte()
        self.protocol = stream.read_byte()
        reserved = stream.read(6)

class Pointer:
    def __init__(self, stream:Stream):
        self.id = stream.read_short()
        self.length = stream.read_int()
        self.index = stream.read_int()

class Tag:
    def __init__(self, stream:Stream):
        self.tag = stream.read_byte()
        self.length = stream.read_short()

        if self.length > 0:
            self.data = stream.read(self.length)

class LeadIdentification:
    def __init__(self, stream:Stream):
        self.start = stream.read_int()
        self.end = stream.read_int()
        self.id = stream.read_byte()

    def __str__(self):
        return '{0} ({1})'.format(self.id, self.sample_count())

    def sample_count(self):
        return self.end - self.start + 1

class Section:
    def __init__(self, header:Header, pointers:list[Pointer]=[]):
        self.header = header
    
    @staticmethod
    def parse(stream:Stream, pointer:Pointer=None):
        if pointer is not None:
            stream.jump(pointer.index - 1)
        header = Header(stream)
        
        if header.id == 0:
            return Section0(header, stream)
        elif header.id == 1:
            return Section1(header, stream)
        else:
            return SectionUnknown(header, stream)

class Section0(Section):
    def __init__(self, header:Header, stream:Stream):
        super().__init__(header)

        self.pointers = []

        count = (header.length - 120 - 16) // 10 + 12
        for _ in range(0, count):
            pointer = Pointer(stream)
            if pointer.length == 0:
                continue
            self.pointers.append(pointer)
    
    def has_section(self, section_id):
        pointer = self.pointer_for_section(section_id)
        if pointer is None:
            return False
        return pointer.length > 0

    def pointer_for_section(self, section_id):
        for pointer in self.pointers:
            if pointer.id == section_id:
                return pointer
        return None

class Section1(Section):
    def __init__(self, header:Header, stream:Stream):
        super().__init__(header)
        
        self.tags = []
        length = header.length - 16
        while length > 0:
            tag = Tag(stream)
            length -= tag.length
            self.tags.append(tag)



class SectionUnknown(Section):
    def __init__(self, header:Header, stream:Stream):
        super().__init__(header)

        self.data = stream.read(header.length - 16)

class SCPFile:
    def __init__(self, stream:Stream):
        self.crc = stream.read_short()
        self.length = stream.read_int()
        self.sections = []
        self.section0 = Section.parse(stream)
        self.sections.append(self.section0)
        for pointer in self.section0.pointers:
            self.sections.append(Section.parse(stream, pointer))

class HeartbeatAria:
    def __init__(self, path:str):
        file = SCPFile(Stream(path))
        if file.section0.has_section(1):
            for t in file.sections[2].tags:
                print(t.tag)
        

HeartbeatAria("Signal.scp")