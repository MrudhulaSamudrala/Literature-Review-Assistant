"""Small text PDFs for ingestion tests."""

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_bytes(pages: list[str], title: str | None = None, author: str | None = None) -> bytes:
    writer = PdfWriter()
    for text in pages:
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[-1]
        stream = DecodedStreamObject()
        stream.set_data(_text_operators(text).encode("latin-1", errors="replace"))
        page[NameObject("/Contents")] = stream
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
    if title:
        writer.add_metadata({"/Title": title})
    if author:
        writer.add_metadata({"/Author": author})
    output = __import__("io").BytesIO()
    writer.write(output)
    data = output.getvalue()
    PdfReader(__import__("io").BytesIO(data))
    return data


def _text_operators(text: str) -> str:
    lines = text.splitlines() or [text]
    commands = ["BT", "/F1 12 Tf", "72 740 Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append("0 -18 Td")
        commands.append(f"({_escape(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
