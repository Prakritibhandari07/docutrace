from app.evidence_verifier import verify_answer


answer = """
Packet switching is a connectionless network switching technique. [E1]

Each packet contains a header and a payload. [E1]

The header contains addressing information. [E1]
"""


evidence = [
    {
        "id": "E1",
        "document": "sWITCHING.pdf",
        "page": 7,
        "text": """
        Packet switching is a connectionless network switching
        technique. The message is divided into packets.
        Each packet has two parts: a header and a payload.
        The header contains the addressing information.
        """
    }
]


results = verify_answer(
    answer,
    evidence
)


for result in results:

    print("\nClaim:")
    print(result["claim"])

    print("Evidence:")
    print(result["evidence_ids"])

    print("Status:")
    print(result["status"])

    print("Reason:")
    print(result["reason"])