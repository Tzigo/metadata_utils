from typing import BinaryIO


def __load_header_data(file: BinaryIO) -> dict:
    """
    Extracts header data from given stream
    :param file: File stream to read from
    :return: header data dict from json
    """

    # local imports
    import json

    # header size (u64) from first 8 bytes of file (saved as little endian)
    header_size = file.read(8)

    # convert read bytes to int
    header_size_int = int.from_bytes(header_size, byteorder='little')

    # read in and load header data
    header = file.read(header_size_int)

    return json.loads(header)


def load_metadata(file_name: str) -> dict:
    """
    Loads file from given file name and extracts metadata

    **Contains no error handling!**
    **You** have to ensure that the file exists and that it is a .safetensors file
    :param file_name: name of the file to load
    :return: metadata dict from json
    """

    # local imports
    import json

    # open file as readonly in binary mode
    file = open(file_name, 'rb')

    # extract header data from file
    header_data = __load_header_data(file)

    metadata = {}

    # extract metadata from header
    for key, value in header_data.get("__metadata__", {}).items():
        metadata[key] = value
        if isinstance(value, str) and value[0:1] == '{':
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                pass

    return metadata


def write_metadata(file_name: str, metadata: dict, new_file_name: str):
    """
    Creates create a new safetensor from given file name and writes it with given metadata into a new file.

    Structure of a .safetensors file:
    8 bytes - byte size of header (n), excluding this size metric
    n bytes - header data in utf-8 encoded json format
    ? bytes (optional?) - padding of space characters (each 1 byte) to next 8 byte border
    ? bytes - tensor data

    Additional information:
     - (header size [in bytes] + padding size [in bytes]) % 8 == 0
    - Metadata is saved in the header json at the key "__metadata__"

    **Contains no error handling!**
    **You** have to ensure that the file exists and that it is a .safetensors file
    :param file_name: name of the original file
    :param metadata: metadata dict
    :param new_file_name: name for the new file
    """

    # local imports
    import json
    import io

    # open file as readonly in binary mode
    with open(file_name, 'rb') as content:
        # extract header data and header size from file
        header_data = __load_header_data(content)

        # overwrite metadata in header
        header_data['__metadata__'] = metadata

        # convert changed header data back to binary string
        new_header_data = json.dumps(header_data).encode('utf-8')

        # open new file to write to
        with open(new_file_name, 'wb') as new_file:
            # calculate space count at the end to next 8 byte length
            buff_spaces = 8 - (len(new_header_data) % 8)

            # calculate new tensor offset
            offset = len(new_header_data) + buff_spaces

            # write new tensor offset into file
            new_file.write(offset.to_bytes(8, 'little'))

            # write new header data into file
            new_file.write(new_header_data)

            # write buffer spaces to next 8 byte length
            new_file.write(b' ' * buff_spaces)

            # load in tensors from original file and write them into new file, chunk by chunk
            chunk_size = io.DEFAULT_BUFFER_SIZE
            chunk = content.read(chunk_size)

            while chunk:
                new_file.write(chunk)
                chunk = content.read(chunk_size)
