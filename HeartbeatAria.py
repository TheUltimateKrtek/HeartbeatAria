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
        self.data = []

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
    _PARSERS = {}
    def __init__(self, header:Header, stream:Stream=None):
        self.header = header
    
        @staticmethod
        def register_parser(id, cls):
            Section._PARSERS[id] = cls

        @staticmethod
        def section_parser(id):
            """Decorator to register a class as a section parser for a given section ID."""
            def decorator(cls):
                Section.register_parser(id, cls)
                return cls
            return decorator
    
    @staticmethod
    def parse(stream:Stream, pointer:Pointer, file:SCPFile):
        if pointer is not None:
            stream.jump(pointer.index - 1)
        header = Header(stream)
        
        Parser = Section._PARSERS.get(header.id)
        if Parser is not None:
            return Parser(header, stream, file)
        else:
            return SectionUnknown(header, stream)
    
    @staticmethod
    def register_parser(id, cls):
        Section._PARSERS[id] = cls

    @staticmethod
    def parser(id):
        """Decorator to register a class as a section parser for a given section ID."""
        def decorator(cls):
            Section.register_parser(id, cls)
            return cls
        return decorator

@Section.parser(0)
class Section0(Section):
    '''Section 0 contains the pointers to each of the sections in the SCP file.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
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

@Section.parser(1)
class Section1(Section):
    '''Section 1 is the Table of Contents (ToC) which contains the labels for leads,
        ECG signals, and related metadata.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)
        
        self.tags = []
        length = header.length - 16
        while length > 0:
            tag = Tag(stream)
            length -= tag.length
            self.tags.append(tag)

@Section.parser(2)
class Section2(Section):
    '''Section 2 contains data on the number of leads, their types, and other lead-related information.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        self.huffman_table_count = stream.read_short()
        self.code_struct_count = stream.read_short()

@Section.parser(3)
class Section3(Section):
    '''Section 3 contains information about reference beats, flags, and the number of leads.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        self.lead_count = stream.read_short()
        self.tags = stream.read_short()
        
        self.reference_beat_substring_tag = bool(self.tags >> 1 & 1)
        
        lead_count = self.tags >> 3 & 0b1111
        self.leads = []
        for _ in range(0, lead_count):
            lead = LeadIdentification(stream)
            self.leads.append(lead)

@Section.parser(4)
class Section4(Section):
    '''Section 4 contains information about the reference beat, fiducial points, and QRS complexes.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        self.reference_beat_type = stream.read_short()
        self.fiducal_point_sample_number = stream.read_short()
        self.qrs_count = stream.read_short()

@Section.parser(5)
class Section5(Section):
    '''Section 5 contains data related to the ECG signal samples for each lead.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        number_of_leads = len(file.get_section(3).leads)

        self.average_measurement = stream.read_short()
        self.sample_time_interval = stream.read_short()
        self.sample_encoding = stream.read_byte()
        reserved = stream.read_byte()

        lead_lengths = []
        for _ in range(0, number_of_leads):
            lead_lengths.append(stream.read_short())
        
        self.samples = []
        for length in lead_lengths:
            samples = []
            sample_count = length / 2
            while sample_count > 0:
                samples.append(stream.read_short())
                sample_count -= 1
            self.samples.append(samples)

@Section.parser(6)
class Section6(Section):
    '''Section 6 contains metadata or other data related to the ECG signal. This could include the number of signals, their processing parameters, etc.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)
        
        number_of_leads = len(file.get_section(3).leads)
        self.average_measurement = stream.read_short()
        self.sample_time_interval = stream.read_short()
        self.sample_encoding = stream.read_byte()
        self.bimodal_compression = stream.read_byte()

        lead_lengths = []
        for _ in range(0, number_of_leads):
            lead_lengths.append(stream.read_short())
        
        self.samples = []
        for length in lead_lengths:
            samples = []
            sample_count = length / 2
            while sample_count > 0:
                samples.append(stream.read_short())
                sample_count -= 1
            self.samples.append(samples)

@Section.parser(7)
class Section7(Section):
    '''Section 7 contains the actual signal data.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        self.reference_count = stream.read_byte()
        self.pace_count = stream.read_byte()
        self.rr_interval = stream.read_short()
        self.pp_interval = stream.read_short()
        
        self.pace_times = []
        self.pace_amplitudes = []
        self.pace_types = []
        self.pace_sources = []
        self.pace_indexes = []
        self.pace_widths = []
        for i in range(0, self.pace_count):
            self.pace_times.append(stream.read_short())
            self.pace_amplitudes.append(stream.read_short())
        
        for i in range(0, self.pace_count):
            self.pace_types.append(self.reader.read_byte())
            self.pace_sources.append(self.reader.read_byte())
            self.pace_indexes.append(self.reader.read_short())
            self.pace_widths.append(self.reader.read_short())

class SectionUnknown(Section):
    '''Unknown section representation.'''
    def __init__(self, header:Header, stream:Stream):
        super().__init__(header)

        self.data = stream.read(header.length - 16)

class SCPFile:
    def __init__(self, stream:Stream):
        self.crc = stream.read_short()
        self.length = stream.read_int()
        self.sections = []
        self.section0 = Section.parse(stream, None, self)
        self.sections.append(self.section0)
        for pointer in self.section0.pointers:
            self.sections.append(Section.parse(stream, pointer, self))
        
    def get_section(self, id:int) -> Section:
        for section in self.sections:
            if section.header.id == id:
                return section
        return None

class HeartbeatAria:
    def __init__(self, path:str):
        file = SCPFile(Stream(path))
        print(file.sections)
        

HeartbeatAria("Signal.scp")