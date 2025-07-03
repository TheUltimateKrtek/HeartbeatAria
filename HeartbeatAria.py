from __future__ import annotations

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "scipy"])

import struct
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import subprocess
import matplotlib
import scipy.signal

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
    
    def read(self, n=1):
        res = self._file.read(n)
        return res
    
    def read_byte(self):
        data = self.read(1)
        return struct.unpack("<B", data)[0]
    
    def read_short(self):
        data = self.read(2)
        return struct.unpack("<H", data)[0]
    
    def read_int(self):
        data = self.read(4)
        return struct.unpack("<I", data)[0]
    
    def read_long(self):
        data = self.read(6)
        return struct.unpack("<Q", data)[0]
    
    def read_str(self, n=1):
        data = self.read(n)
        return data.decode('iso-8859-1')
    
    def read_bits(self, little=False):
        byte = self._file.read(1)
        if not byte:  # End of file
            return []
        if little:
            return [(byte[0] >> i) & 1 for i in range(8)]
        return [(byte[0] >> i) & 1 for i in range(7, -1, -1)]  # Extract bits (MSB first)

class HuffmanNode:
    def __init__(self, value=None):
        self.path = []
        self.value = value
        self.decoded = 0
    
    @property
    def is_leaf(self) -> bool:
        return len(self.path) < 2
    
    @property
    def zero(self) -> HuffmanNode|None:
        return None if self.is_leaf else self.path[0]
    
    @property
    def one(self) -> HuffmanNode|None:
        return None if self.is_leaf else self.path[1]
    
    @zero.setter
    def zero(self, value):
        if self.is_leaf:
            self.path = [value, None]
        else:
            self.path[0] = value
    
    @one.setter
    def one(self, value):
        if self.is_leaf:
            self.path = [None, value]
        else:
            self.path[1] = value
    
    def read(self, reader:HuffmanReader):
        if self.is_leaf:
            if self.value is not None:
                return self.value
            read = self.read_leaf(reader)
            self.decoded += 1
            return read
        path = reader.read_bits(1)[0]
        return self.path[path].read(reader)
    
    def read_leaf(self, reader:HuffmanReader):
        return None
    
    def describe_tree(self, path="", level=0):
        if not self.is_leaf:
            print("".join(["|  " for _ in range(level - 1)]) + str(path))
            self.zero.describe_tree(path + str(0), level + 1)
            self.one.describe_tree(path + str(1), level + 1)
        else:
            print("".join(["|  " for _ in range(level - 1)]) + str(path) + " " + str(self.value) + " " + str(self.decoded))

class DefaultSCPHuffmanLeafNode(HuffmanNode):
    def __init__(self, bits:int=0):
        super().__init__()
        self.bits = bits
    
    def read_leaf(self, reader:HuffmanReader):
        bits = reader.read_bits(self.bits)
        value = 0
        for i, b in enumerate(bits):
            value = (value << 1) | b
            # value = value | (b << i)
        if self.bits == 8:
            return struct.unpack("<b", struct.pack("<B", value))[0]
        else:
            return struct.unpack("<h", struct.pack("<H", value))[0]
    
    def describe_tree(self, path="", level=0):
        print("".join(["|  " for _ in range(level - 1)]) + str(path) + " " + str(self.bits) + " " + str(self.decoded))
    
    @staticmethod
    def create_tree():
        root = HuffmanNode()
        root.zero = HuffmanNode(0)
        root.one = HuffmanNode()
        current = root.one
        for i in range(8):
            current.zero = HuffmanNode()
            current.zero.zero = HuffmanNode(i + 1)
            current.zero.one = HuffmanNode(- i - 1)
            current.one = HuffmanNode()
            current = current.one
        current.zero = DefaultSCPHuffmanLeafNode(8)
        current.one = DefaultSCPHuffmanLeafNode(16)
        # root.describe_tree()
        return root

class HuffmanReader:
    def __init__(self, stream:Stream, tree:HuffmanNode):
        self.stream = stream
        self.tree = tree
        self.buffer = []
        self.bytes_read = 0
        self.values_read = 0
        # tree.describe_tree()
    
    def read_bits(self, n=1):
        while len(self.buffer) < n:
            for b in self.stream.read_bits():
                self.buffer.append(b)
            self.bytes_read += 1
        bits = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return bits
    
    def read(self):
        self.values_read += 1
        r = self.tree.read(self)
        return r
    
    def reset(self):
        self.buffer = []
        self.bytes_read = 0
        self.values_read = 0

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
    
    def __str__(self):
        return f"Pointer: {self.id}, Length: {self.length}, Index: {self.index}"

class Tag:
    def __init__(self, stream:Stream):
        self.tag = stream.read_byte()
        self.length = stream.read_short()
        self.data = []

        if self.length > 0:
            self.data = stream.read(self.length)
    
    def __str__(self):
        return f"Tag: {self.tag}, Length: {self.length}, Data: {self.data if len(self.data) < 20 else str(self.data[:20]) + '...'}"

class LeadIdentification:
    NAMES = [
        "unspecified", "I", "II", "V1", "V2", "V3", "V4",
        "V5", "V6", "V7", "V2R", "V3R", "V4R", "V5R", "V6R",
        "V7R", "X", "Y", "Z", "CC5", "CM5", "left arm",
        "right arm", "left leg", "I", "E", "C", "A", "M",
        "F", "H", "I-cal", "II-cal", "V1-cal", "V2-cal",
        "V3-cal", "V4-cal", "V5-cal", "V6-cal", "V7-cal",
        "V2R-cal", "V3R-cal", "V4R-cal", "V5R-cal", "V6R-cal",
        "V7R-cal", "X-cal", "Y-cal", "Z-cal", "CC5-cal",
        "CM5-cal", "left arm-cal", "right arm-cal",
        "left leg-cal", "I-cal", "E-cal", "C-cal", "A-cal",
        "M-cal", "F-cal", "H-cal", "III", "aVR", "aVL", "aVF",
        "-aVR", "V8", "V9", "V8R", "V9R", "D (Nehb-Dorsal)",
        "A (Nehb-Anterior)", "J (Nehb-Inferior)",
        "Defibrillator lead: anterior-lateral",
        "External pacing lead: anterior-posterior",
        "A1 (Auxiliary unipolar lead 1)",
        "A2 (Auxiliary unipolar lead 2)",
        "A3 (Auxiliary unipolar lead 3)",
        "A4 (Auxiliary unipolar lead 4)",
        "V8-cal", "V9-cal", "V8R-cal", "V9R-cal",
        "D-cal (cal for Nehb – Dorsal)",
        "A-cal (cal for Nehb – Anterior)",
        "J-cal (cal for Nehb – Inferior)"
    ]
    def __init__(self, stream:Stream):
        self.start = stream.read_int()
        self.length = stream.read_int()
        self.id = stream.read_byte()

    def __str__(self):
        return f'Lead: {self.name} ({self.length})'
    
    @property
    def name(self) -> str:
        return LeadIdentification.NAMES[self.id] if self.id > 0 and self.id < len(LeadIdentification.NAMES) else LeadIdentification.NAMES[0]

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

        print("Section 0:")
        print(f"    Protocol: {self.header.protocol}, Version: {self.header.version}")
        print(f"    Length according to header: {self.header.length}")
        print(f"        {len(self.pointers)} sections:")
        for p in self.pointers:
            print(f"    {p}")

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
        
        print("Section 1:")
        print(f"    Length according to header: {self.header.length}")
        for t in self.tags:
            print(f"    Tag: {t.tag}, Length: {t.length}, Data: {t.data if len(t.data) < 20 else str(t.data[:20]) + "..."}")
    
    def get_tag_contents(self, id:int) -> bytes|list[bytes]:
        rl = []
        for tag in self.tags:
            if tag.tag == id:
                rl.append(tag.data)
        if len(rl) == 0: return None
        if len(rl) == 1: return rl[0]
        return rl

@Section.parser(2)
class Section2(Section):
    '''Section 2 contains data on the number of leads, their types, and other lead-related information.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        self.huffman_encoding_type = stream.read_short()
        
        print("Section 2:")
        print(f"    Length according to header: {self.header.length}")
        print(f"    Huffman table count: {self.huffman_encoding_type}")
        if self.huffman_encoding_type != 19999:
            raise NotImplementedError("Only encoding type 19999 is implemented.")

@Section.parser(3)
class Section3(Section):
    '''Section 3 contains information about reference beats, flags, and the number of leads.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        # Read the lead count (byte 0)
        self.lead_count = stream.read_byte()
        
        # Read the lead flags (byte 1) - not needed for now
        self.lead_flags = stream.read_byte()
        self.subtract_encoding = ((self.lead_flags >> 0) & 1) == 0
        self.all_simultaniously_recorded = ((self.lead_flags >> 2) & 2) == 0
        self.simultanious_count = (self.lead_flags >> 3) & 0x1F
        
        # Calculate the number of leads based on the section length
        lead_count = (header.length - 16) // 9
        
        self.leads = []
        
        # Iterate over the leads and read the 9-byte chunks
        for _ in range(lead_count):
            lead = LeadIdentification(stream)
            self.leads.append(lead)
        
        # Output the lead data
        print("Section 3:")
        print(f"    Length according to header: {self.header.length}")
        print(f"    Subtract encoding: {self.subtract_encoding}")
        print(f"    All simultaniously recorded: {self.all_simultaniously_recorded}")
        print(f"    Simultanious count: {self.simultanious_count}")
        print(f"    Count: {lead_count}")
        print("Leads: " + ", ".join([l.name for l in self.leads]))

@Section.parser(4)
class Section4(Section):
    '''Section 4 contains information about the reference beat, fiducial points, and QRS complexes.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        self.reference_beat_type = stream.read_short()
        self.fiducal_point_sample_number = stream.read_short()
        self.qrs_count = stream.read_short()

        self.qrs = []
        for i in range(self.qrs_count):
            type = stream.read_short()
            index = stream.read_int()
            fiducial_point = stream.read_int()
            end = stream.read_int()
            self.qrs.append((type, index, fiducial_point, end)) # TODO

        print("Section 4:")
        print(f"    Length according to header: {self.header.length}")
        print(f"    Reference beat type: {self.reference_beat_type}")
        print(f"    Fiducal Point Sample Number: {self.fiducal_point_sample_number}")
        print(f"    Qrs count: {self.qrs_count}")

@Section.parser(5)
class Section5(Section):
    '''Section 5 contains data related to the ECG signal samples for each lead.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        number_of_leads = len(file.get_section(3).leads)

        self.amplitude_multiplier = stream.read_short()
        self.sample_time_interval = stream.read_short()
        self.difference_encoding = stream.read_byte()
        reserved = stream.read_byte()

        lead_lengths = []
        for _ in range(0, number_of_leads):
            lead_lengths.append(stream.read_short())
        
        section2 = file.get_section(2)
        reader = None
        if not section2:
            reader = stream
        elif section2.huffman_encoding_type == 19999:
            reader = HuffmanReader(stream, DefaultSCPHuffmanLeafNode.create_tree())
        
        self.samples = []
        for length in lead_lengths:
            samples = []
            if not section2:
                for i in range(length // 2):
                    samples.append(reader.read_short())
            else:
                reader.bytes_read = 0
                reader.samples_read = 0
                while reader.bytes_read < length:
                    samples.append(reader.read())
            self.samples.append(samples)

        print("Section 5:")
        print(f"    Length according to header: {self.header.length}")
        print(f"    Amplitude multiplier: {self.amplitude_multiplier}")
        print(f"    Sample time interval: {self.sample_time_interval}")
        print(f"    Difference encodeing: {self.difference_encoding}")
        print(f"    Lead lenths: {lead_lengths}")
        for l, lead, samples in zip(lead_lengths, file.get_section(3).leads, self.samples):
            print(f"    Lead {lead.id}: {lead.name}, {l} bytes read, {len(samples)} samples")

@Section.parser(6)
class Section6(Section):
    '''Section 6 contains metadata or other data related to the ECG signal. This could include the number of signals, their processing parameters, etc.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        number_of_leads = len(file.get_section(3).leads)
        self.amplitude_multiplier = stream.read_short()
        self.sample_time_interval = stream.read_short()
        self.difference_encoding = stream.read_byte()
        self.bimodal_compression = stream.read_byte()

        lead_lengths = []
        for _ in range(0, number_of_leads):
            lead_lengths.append(stream.read_short())
        
        section2 = file.get_section(2)
        reader = None
        if not section2:
            reader = stream
        elif section2.huffman_encoding_type == 19999:
            reader = HuffmanReader(stream, DefaultSCPHuffmanLeafNode.create_tree())

        self.samples = []
        for i in range(len(lead_lengths)):
            length = lead_lengths[i]
            samples = []
            if not section2:
                for i in range(length // 2):
                    samples.append(reader.read_short())
            else:
                reader.bytes_read = 0
                reader.samples_read = 0
                # while reader.bytes_read < length:
                while reader.values_read < file.get_section(3).leads[i].length:
                    samples.append(reader.read())
            self.samples.append(samples)

            # Flush buffer
            reader.reset()

        print("Section 6:")
        print(f"    Length according to header: {self.header.length - 16 - 2 * len(lead_lengths)} ({sum(lead_lengths)} read)")
        print(f"    Amplitude multiplier: {self.amplitude_multiplier}")
        print(f"    Sample time interval: {self.sample_time_interval}")
        print(f"    Difference encodeing: {self.difference_encoding}")
        print(f"    Bimodal compression: {self.bimodal_compression}")
        print(f"    Lead lenths: {lead_lengths}")
        for l, lead, samples in zip(lead_lengths, file.get_section(3).leads, self.samples):
            print(f"    Lead {lead.id}: {lead.name}, {lead.length} samples expected, {l} bytes read, {len(samples)} samples")
        
        # print(reader.tree.describe_tree())

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

        print("Section 7:")
        print(f"    Length according to header: {self.header.length}")
        print(f"    Reference count: {self.reference_count}")
        print(f"    Pace count: {self.pace_count}")
        print(f"    RR interval: {self.rr_interval}")
        print(f"    PP interval: {self.pp_interval}")
        print(f"    Pace length: {len(self.pace_times)}")

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
        section0 = Section.parse(stream, None, self)
        for pointer in section0.pointers:
            self.sections.append(Section.parse(stream, pointer, self))
        
    def get_section(self, id:int) -> Section:
        for section in self.sections:
            if section.header.id == id:
                return section
        return None

class Measurement:
    class Height:
        CONVERSION_FACTORS = [
            [1, 1, 1, 1],
            [1, 1, 1/2.54, 10],
            [1, 2.54, 1, 25.4],
            [1, 0.1, 1/25.4, 1]
        ]
        NAMES = ["", "cm", "inch", "mm"]

        def __init__(self, value:int|float, units:int):
            self._value = value
            self._units = units
        
        def __str__(self):
            return self.value + ((" " + self.units) if self.units else "")
        
        @staticmethod
        def parse(data) -> Measurement.Height|None:
            if data is None: return None
            return Measurement.Height(int.from_bytes(data[:2], data[2]))
        
        @property
        def value(self) -> float:
            return self._value
        
        @property
        def units_id(self) -> int:
            return self._units if self._units >= 0 and self._units < len(Measurement.Height.NAMES) else 0

        @property
        def units(self) -> str:
            return Measurement.Height.NAMES[self.units_id]
        
        @property
        def is_specified(self) -> bool:
            return self._value != 0 or self._units != 0
        
        def convert(self, units:int|str):
            if isinstance(units, str):
                if units not in Measurement.Height.NAMES:
                    units = 0
                else:
                    units = Measurement.Height.NAMES.index(units.lower())

            units_from = self.units if self.units >= 0 and self.units < len(Measurement.Height.NAMES) else 0
            units_to = units if units >= 0 and units < len(Measurement.Height.NAMES) else 0
            return Measurement.Height(self.value * Measurement.Height.CONVERSION_FACTORS[units_from][units_to], units_to)
    
    class Weight:
        CONVERSION_FACTORS = [
            [1, 1, 1, 1, 1],                     # Unspecified (copy value)
            [1, 1, 1000, 2.20462, 35.274],       # Kilograms
            [1, 0.001, 1, 0.00220462, 0.035274], # Grams
            [1, 0.453592, 453.592, 1, 16],       # Pounds
            [1, 0.0283495, 28.3495, 0.0625, 1]   # Ounces
        ]
        NAMES = ["", "kg", "g", "lbs", "oz"]

        def __init__(self, value:int|float, units:int):
            self._value = value
            self._units = units
        
        def __str__(self):
            return self.value + ((" " + self.units) if self.units else "")
        
        @staticmethod
        def parse(data) -> Measurement.Weight|None:
            if data is None: return None
            return Measurement.Weight(int.from_bytes(data[:2], data[2]))
        
        
        @property
        def value(self) -> float:
            return self.value
        
        @property
        def units_id(self) -> int:
            return self._units if self._units >= 0 and self._units < len(Measurement.Height.NAMES) else 0
        
        @property
        def units(self) -> str:
            return Measurement.Weight.NAMES[self.units_id]
        
        @property
        def is_specified(self) -> bool:
            return self._value != 0 or self._units != 0
        
        def convert(self, units:str|int):
            if isinstance(units, str):
                if units not in Measurement.Weight.NAMES:
                    units = 0
                else:
                    units = Measurement.Weight.NAMES.index(units.lower())
                
            units_from = self._units if self._units >= 0 and self._units < len(Measurement.Weight.NAMES) else 0
            units_to = units if units >= 0 and units < len(Measurement.Weight.NAMES) else 0
            return Measurement.Weight(self.value * Measurement.Weight.CONVERSION_FACTORS[units_from][units_to], units_to)
    
    class Timestamp:
        def __init__(self, day:tuple, time:tuple=None):
            self.year, self.month, self.day = day
            self.hour, self.minute, self.second = time if time is not None else (None, None, None)
        
        @property
        def is_time_specified(self):
            return self.hour is not None

class Patient:
    #TODO: Check if correct
    def __init__(self, section:Section1):
        self.last_name = Patient._parse_str(section.get_tag_contents(0))
        self.first_name = Patient._parse_str(section.get_tag_contents(1))
        self.id = Patient._parse_str(section.get_tag_contents(2))
        self.second_last_name = Patient._parse_str(section.get_tag_contents(3))

        self.age, self.age_units = Patient._unpack(section.get_tag_contents(4), [2, 1])
        self.age_units = ["unspecified", "years", "months", "weeks", "days", "hours"][self.age_units] if self.age_units and self.age_units >= 0 and self.age_units <= 5 else "unspecified"
        self.age_specified = self.age != 0 and self.age_units != 0

        self.birthday = Measurement.Timestamp(self._unpack(section.get_tag_contents(5), [2, 1, 1]))

        self.height = Measurement.Height.parse(section.get_tag_contents(6))
        self.weight = Measurement.Weight.parse(section.get_tag_contents(7))

        self.sex = Patient._unpack(section.get_tag_contents(8))
        if self.sex is None: self.sex = 0
        self.sex = ["not known", "male", "female"][self.sex] if self.sex < 3 else "unspecified"

        self.race_id = Patient._unpack(section.get_tag_contents(9))
        if self.race_id is None: self.race_id = 0
        if self.race_id < 4: self.race = ["unspecified", "caucasian", "black", "oriental"][self.race_id]
        else: self.race = "other"
        
        self.classification = section.get_tag_contents(10) 
        # TODO

        self.systolic_blood_pressure = Patient._unpack(section.get_tag_contents(11))
        self.diatolic_blood_pressure = Patient._unpack(section.get_tag_contents(12))

        self.diagnosis = Patient._parse_str(section.get_tag_contents(13))

        self.acquiring_device_machine_id = section.get_tag_contents(14) # TODO
        self.analyzing_device_machine_id = section.get_tag_contents(15) # TODO

        self.acquiring_institution = Patient._parse_str(section.get_tag_contents(16))
        self.analyzing_institution = Patient._parse_str(section.get_tag_contents(17))
        self.acquiring_department = Patient._parse_str(section.get_tag_contents(18))
        self.analyzing_department = Patient._parse_str(section.get_tag_contents(19))

        self.referring_physician = Patient._parse_str(section.get_tag_contents(20))
        self.confirming_physician = Patient._parse_str(section.get_tag_contents(21))
        self.technician = Patient._parse_str(section.get_tag_contents(22))
        self.room = Patient._parse_str(section.get_tag_contents(23))
        
        self.emergency = Patient._unpack(section.get_tag_contents(24), default=0)

        self.time_of_aquisition = Measurement.Timestamp(
            Patient._unpack(section.get_tag_contents(25), [2, 1, 1]),
            Patient._unpack(section.get_tag_contents(26), [1, 1, 1])
        )
        self.baseline_filter = Patient._unpack(section.get_tag_contents(27))
        self.lowpass_filter = Patient._unpack(section.get_tag_contents(28))
        self.filter_bit_map = Patient._unpack(section.get_tag_contents(29))
        
        self.comments = section.get_tag_contents(30)
        self.comments = [Patient._parse_str(b) for b in self.comments] if self.comments else None
        self.sequence_number = Patient._parse_str(section.get_tag_contents(31))
        
        self.medical_history = Patient._unpack(section.get_tag_contents(32), default=0)

        self.electrode_placement, self.electrode_system = Patient._unpack(section.get_tag_contents(33), [1, 1])

        self.timezone, self.timezone_index, self.timezone_description = Patient._unpack(section.get_tag_contents(24), [2, 2], last_is_str_until_end=True)

        self.medical_text_history = Patient._parse_str(section.get_tag_contents(35))

    @staticmethod
    def _unpack(data:bytes, format=None, default=None, last_is_str_until_end=False):
        if data is None:
            if format is None:
                return default
            return [None for _ in range(len(format) + int(last_is_str_until_end))]
        if format is None:
            format = [len(data)]
        if isinstance(data, list):
            if len(data) == 0:
                return default
            data = data[0]
        rl = []
        start = 0
        for f in format:
            rl.append(int.from_bytes(data[start:start + f]))
        if last_is_str_until_end:
            rl.append(Patient._parse_str(data[start:]))
        
        if len(rl) == 1: return rl[0]
        return rl
    
    def _parse_str(data, default=None):
        if data is None:
            return default
        return data.decode('iso-8859-1')

class Signal:
    def __init__(self, name:str, signal:list, sample_time_interval:int, amplitude_multiplier:int=1, difference_encoding:int=0):
        self.signal = np.array(signal)
        if difference_encoding == 1:
            for i in range(1, len(self.signal)):
                self.signal[i] += self.signal[i - 1]
        elif difference_encoding == 2:
            orig = self.signal.copy()
            for i in range(2, len(orig)):
                orig[i] = orig[i] + 2 * orig[i - 1] - orig[i - 2]
            self.signal = orig
        
        self.amplitude_multiplier = amplitude_multiplier * 0.000000001
        self.name = name
        self.sample_time_interval = sample_time_interval
    
    def detrend(self) -> 'Signal':
        """
        Returns a new Signal with the linear trend removed.
        """
        detrended_signal = scipy.signal.detrend(self.signal)
        return Signal(self.name, detrended_signal, self.sample_time_interval, self.amplitude_multiplier / 0.000000001, 0)
    
    def smooth_outliers(self, window_size=5, threshold=3) -> 'Signal':
        """
        Returns a new Signal with outliers at the beginning and end smoothed out.
        Outliers are defined as points more than `threshold` standard deviations from the mean
        in the first and last `window_size` samples.
        """
        signal = self.signal.copy()
        n = len(signal)
        # Smooth start
        start_window = signal[:window_size]
        mean = np.mean(start_window)
        std = np.std(start_window)
        for i in range(window_size):
            if abs(signal[i] - mean) > threshold * std:
                # Replace with mean of next window_size values (or as many as available)
                next_vals = signal[i+1:i+1+window_size]
                if len(next_vals) > 0:
                    signal[i] = np.mean(next_vals)
                else:
                    signal[i] = mean
        # Smooth end
        end_window = signal[-window_size:]
        mean = np.mean(end_window)
        std = np.std(end_window)
        for i in range(n - window_size, n):
            if abs(signal[i] - mean) > threshold * std:
                prev_vals = signal[max(0, i-window_size):i]
                if len(prev_vals) > 0:
                    signal[i] = np.mean(prev_vals)
                else:
                    signal[i] = mean
        return Signal(self.name, signal, self.sample_time_interval, self.amplitude_multiplier / 0.000000001, 0)
    
    def aggressive_smooth_outliers(self, window_size=10, threshold=1) -> 'Signal':
        """
        Returns a new Signal with aggressive outlier smoothing based on the first difference.
        Outliers are detected where the difference between consecutive samples exceeds the threshold.
        Outliers are replaced by the median of a window around them.
        """
        signal = self.signal.copy()
        n = len(signal)
        diffs = np.diff(signal, prepend=signal[0])

        # Forward pass
        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            window = signal[start:end]
            diff_window = diffs[start:end]
            median = np.median(window)
            std_diff = np.std(diff_window)
            if std_diff > 0 and abs(diffs[i]) > threshold * std_diff:
                signal[i] = median

        # Backward pass
        diffs = np.diff(signal, prepend=signal[0])
        for i in reversed(range(n)):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            window = signal[start:end]
            diff_window = diffs[start:end]
            median = np.median(window)
            std_diff = np.std(diff_window)
            if std_diff > 0 and abs(diffs[i]) > threshold * std_diff:
                signal[i] = median

        return Signal(self.name, signal, self.sample_time_interval, self.amplitude_multiplier / 0.000000001, 0)
    
    def fix_edge_outliers(self, edge_size=100, threshold=1, max_derivative=3) -> 'Signal':
        """
        Detects outlier zones at the start and end of the signal, and replaces them
        by extrapolating from the nearest good data using the lowest derivative that fits.
        """
        signal = self.signal.copy()
        n = len(signal)

        def find_outlier_zone(arr, from_start=True):
            window = arr[:edge_size] if from_start else arr[-edge_size:]
            mean = np.mean(window)
            std = np.std(window)
            if std == 0:
                return 0
            if from_start:
                for i in range(edge_size):
                    if abs(arr[i] - mean) > threshold * std:
                        return i
                return edge_size
            else:
                for i in range(edge_size):
                    if abs(arr[-(i+1)] - mean) > threshold * std:
                        return i
                return edge_size

        # Find outlier zones
        start_zone = find_outlier_zone(signal, from_start=True)
        end_zone = find_outlier_zone(signal, from_start=False)

        # Fix start
        if start_zone > 0:
            good_start = start_zone
            for d in range(1, max_derivative+1):
                if good_start + d >= n:
                    break
                # Fit polynomial of degree d to next edge_size points
                x = np.arange(good_start, good_start + edge_size)
                y = signal[good_start:good_start + edge_size]
                coeffs = np.polyfit(x, y, min(d, len(x)-1))
                for i in range(good_start):
                    signal[i] = np.polyval(coeffs, i)
                # Stop at first derivative that fits
                break

        # Fix end
        if end_zone > 0:
            good_end = n - end_zone
            for d in range(1, max_derivative+1):
                if good_end - edge_size < 0:
                    break
                x = np.arange(good_end - edge_size, good_end)
                y = signal[good_end - edge_size:good_end]
                coeffs = np.polyfit(x, y, min(d, len(x)-1))
                for i in range(good_end, n):
                    signal[i] = np.polyval(coeffs, i)
                break

        return Signal(self.name, signal, self.sample_time_interval, self.amplitude_multiplier / 0.000000001, 0)
    
    def remove_edge_outliers(self, edge_size=100, threshold=1) -> 'Signal':
        """
        Detects outlier zones at the start and end of the signal and removes these zones entirely.
        Returns a new Signal with the outlier zones removed.
        """
        signal = self.signal.copy()
        n = len(signal)

        def find_outlier_zone(arr, from_start=True):
            window = arr[:edge_size] if from_start else arr[-edge_size:]
            mean = np.mean(window)
            std = np.std(window)
            if std == 0:
                return 0
            if from_start:
                for i in range(edge_size):
                    if abs(arr[i] - mean) > threshold * std:
                        return i
                return edge_size
            else:
                for i in range(edge_size):
                    if abs(arr[-(i+1)] - mean) > threshold * std:
                        return i
                return edge_size

        # Find outlier zones
        start_zone = find_outlier_zone(signal, from_start=True)
        end_zone = find_outlier_zone(signal, from_start=False)

        # Remove the zones
        new_signal = signal[start_zone:n-end_zone] if end_zone > 0 else signal[start_zone:]

        return Signal(self.name, new_signal, self.sample_time_interval, self.amplitude_multiplier / 0.000000001, 0)
    
    def remove_rogue_beginning(self, threshold=5) -> 'Signal':
        """
        Removes samples at the beginning of the signal that are too low or too high compared to the rest of the signal.
        A sample is considered rogue if it is more than `threshold` standard deviations from the mean of the signal (excluding the first 5%).
        """
        signal = self.signal.copy()
        n = len(signal)
        # Use the main body of the signal (excluding first 5%) to compute mean and std
        start_idx = max(int(n * 0.05), 1)
        main_body = signal[start_idx:]
        mean = np.mean(main_body)
        std = np.std(main_body)
        # Find first index where value is within threshold*std of mean
        idx = 0
        while idx < n and abs(signal[idx] - mean) > threshold * std:
            idx += 1
        # Remove rogue beginning
        new_signal = signal[idx:] if idx > 0 else signal
        return Signal(self.name, new_signal, self.sample_time_interval, self.amplitude_multiplier / 0.000000001, 0)
    
    def attempt_fix(self):
        """
        Applies a series of fixes to the signal:
        1. Detrend the signal.
        2. Aggressively smooth outliers.
        3. Fix edge outliers.
        """
        return self.detrend().smooth_outliers().aggressive_smooth_outliers().fix_edge_outliers().remove_edge_outliers().remove_rogue_beginning().detrend()
    
class Data:
    def __init__(self, path:str):
        file = SCPFile(Stream(path))
        self.patient_info = Patient(file.get_section(1))

        avm5 = file.get_section(5).amplitude_multiplier
        avm6 = file.get_section(6).amplitude_multiplier
        self.data5 = []
        self.data6 = []
        for s5, s6, lead in zip(file.get_section(5).samples, file.get_section(6).samples, file.get_section(3).leads):
            is_first = s6 == file.get_section(6).samples[0]
            self.data5.append(Signal(lead.name, s5, file.get_section(5).sample_time_interval, avm5, file.get_section(5).difference_encoding))
            self.data6.append(Signal(lead.name, s6, file.get_section(6).sample_time_interval, avm6, file.get_section(6).difference_encoding))

        # self.avm6[2] = Signal("III", self.avm6[1].signal - self.avm6[0].signal, file.get_section(5).sample_time_interval, avm5, file.get_section(5).difference_encoding)
        # TODO: Formulas

        for i5 in range(len(self.data5)):
            self.draw_graph(file, signal=i5, section=5)
        for i6 in range(len(self.data6)):
            self.draw_graph(file, signal=i6, section=6)

    def draw_graph(self, file, signal=0, section=6, length=2 ** 30):
        s = self.data6[signal] if section == 6 else self.data5[signal]
        samples = s.attempt_fix().signal[:min(len(s.signal), length)]
        print(f"Displaying {len(samples)} samples.")
        sample_time_interval = s.sample_time_interval * 0.000001
        self.plot_samples(samples, title=f"ECG Signal {s.name}", sample_time_interval=sample_time_interval, section=section, index=signal)

    def plot_samples(self, samples, title="ECG Signal", sample_time_interval=1, section=6, index=0):
        """
            Plots a list of numerical samples.

            :param samples: A list of numerical values representing the signal.
            :param title: Title of the plot.
            :param sampling_rate: The number of samples per second (default is 500 Hz).
        """
        time_axis = [i * sample_time_interval for i in range(len(samples))]

        plt.figure(figsize=(25, 10))
        plt.plot(time_axis, samples, linestyle='-')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (nV)")
        plt.title(title)
        plt.grid()
        plt.savefig(f"graph-{section}-{index}.png")
        print("Graph saved!")

Data("Signal.scp")