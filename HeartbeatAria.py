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
        bytes = self.read(n)
        return self.read(bytes).decode('iso-8859-1')

class Header:
    def __init__(self, stream:Stream):
        self.crc = stream.read_short()
        self.id = stream.read_short()
        self.length = stream.read_int()
        self.version = stream.read_byte()
        self.protocol = stream.read_byte()

class Section(ABC):
    def __new__(cls, stream:Stream):
        header = Header(Stream)
        
        if header.id == 0:
            return Section0(header)
        else:
            return SectionUnknown(header)

        
# TODO: Sections
class Section0(Section):
    def __init__(self, stream:Stream, header:Header):
        self.header = header

        section_count = 12 + header.length - 120 - 16

        self.section_ids = []
        self.section_lengths = []
        self.section_indexes = []
        for _ in range(section_count):
            self.section_ids.append(stream.read_short())
            self.section_lengths.append(stream.read_int())
            self.section_indexes.append(stream.read_int())

class SectionUnknown(Section):
    def __init__(self, stream:Stream, header:Header):
        self.header = header

        self.data = stream.read(header.length - 120 - 16)

class SCPFile:
    def __init__(self, path:str):
        stream = Stream(path)
        
        self.crc = stream.read_short()
        self.length = stream.read_int()
        
        self.sections = []
        self.sections.append(Section(stream))

        # TODO

print(SCPFile("Signal").sections)
    
