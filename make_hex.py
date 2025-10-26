import binascii

def pt_to_hex(pt_file, hex_file):
    with open(pt_file, "rb") as f:
        hexdata = binascii.hexlify(f.read())
    with open(hex_file, "wb") as f:
        f.write(hexdata)

def hex_to_pt(hex_file, pt_file):
    with open(hex_file, "rb") as f:
        bindata = binascii.unhexlify(f.read())
    with open(pt_file, "wb") as f:
        f.write(bindata)

# Example usage
# pt_to_hex("models/best.pt", "models/best.pt.hex")
# hex_to_pt("models/best.pt.hex", "models/best.pt")
