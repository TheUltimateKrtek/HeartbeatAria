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
    
    def read_bits(self):
        byte = self._file.read(1)
        if not byte:  # End of file
            return None
        return [(byte[0] >> i) & 1 for i in range(7, -1, -1)]  # Extract bits (MSB first)

class HuffmanStreamReader:
    def __init__(self, stream:Stream, chunk_size:int):
        self.stream = stream
        self.chunk_size = chunk_size
        self.tree = None
        self.bit_buffer = []

        self.read_tree()
    
    def read_tree(self):
        pass

    def read_bits(self, n=1):
        while len(self.bit_buffer) < n:
            for b in self.stream.read_bits():
                self.bit_buffer.append(b)
        bits = self.bit_buffer[:n]
        self.bit_buffer = self.bit_buffer[n:]
        return bits
    


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
        self.length = stream.read_int()
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

        print("Section 0:")
        print("    Pointer indexes:", [p.index for p in self.pointers])
        print("    Pointer ids:", [p.id for p in self.pointers])
        print("    Pointer lengths:", [p.length for p in self.pointers])

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
        
        # Calculate the number of leads based on the section length
        lead_count = (header.length - 16) // 9
        
        self.leads = []
        
        # Iterate over the leads and read the 9-byte chunks
        for _ in range(lead_count):
            lead = LeadIdentification(stream)
            self.leads.append(lead)
        
        # Output the lead data
        print("Section 3:")
        for l in self.leads:
            print(f"    Lead: {l.id}, Start: {l.start}, Length: {l.length}")

@Section.parser(4)
class Section4(Section):
    '''Section 4 contains information about the reference beat, fiducial points, and QRS complexes.'''
    def __init__(self, header:Header, stream:Stream, file:SCPFile):
        super().__init__(header)

        self.reference_beat_type = stream.read_short()
        self.fiducal_point_sample_number = stream.read_short()
        self.qrs_count = stream.read_short()

        print("Section 4:")
        print(f"    Reference beat type: {self.reference_beat_type}")
        print(f"    Fiducal Point Sample Number: {self.fiducal_point_sample_number}")
        print(f"    Qrs count: {self.qrs_count}")

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

        print("Section 5:")
        print("    TODO")

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

        print("Section 6:")
        print(f"    Lead lenths: {lead_lengths}")
        for lead, sample in zip(file.get_section(3).leads, self.samples):
            print(f"{len(sample)} samples for lead {lead.id}: {sample[20] if len(sample) > 20 else sample}")

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
        print("    TODO")

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
    def __init__(self, section:Section1):
        self.last_name = Patient._parse_str(section.get_tag_contents(0))
        self.first_name = Patient._parse_str(section.get_tag_contents(1))
        self.id = Patient._parse_str(section.get_tag_contents(2))
        self.second_last_name = Patient._parse_str(section.get_tag_contents(3))

        self.age, self.age_units = Patient._unpack(section.get_tag_contents(4), [2, 1])
        self.age_units = ["unspecified", "years", "months", "weeks", "days", "hours"][self.age_units] if self.age_units >= 0 and self.age_units <= 5 else "Unspecified"
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
        
        self.comments = [Patient._parse_str(b) for b in section.get_tag_contents(30)]
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
            return [None for _ in format]
        if format is None:
            format = [len(data)]
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
        
class Data:
    def __init__(self, path:str):
        file = SCPFile(Stream(path))
        self.patient_info = Patient(file.get_section(1))
        tags = file.get_section(3).leads

Data("Signal.scp")