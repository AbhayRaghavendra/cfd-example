from pathlib import Path

from cfd_gnn.download import _extract_file


def test_extract_file_accepts_direct_xml_metadata():
    contents = b"<ctml></ctml>"

    extracted = _extract_file(
        "chem_thermo_tran/bfer.xml",
        Path("/tmp/bfer.xml"),
        contents,
        "text/xml",
    )

    assert extracted == contents
