import requests

upload_url = "http://localhost:9000/omnirag-corpus/uploads/f615d1b4-64a5-436e-8baa-2829cae68444/b9f1438b-b828-43cb-b13d-c6d4324b6f7d.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=omnirag_admin%2F20260909%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260909T142034Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=content-type%3Bhost%3Bx-amz-meta-corpus_id%3Bx-amz-meta-original_filename&X-Amz-Signature=68811cbb2e16a361533f049a7500b0ffd25bc6d304fa19741fcf49cdb847a645"

with open("arpon-kapuria-cv.pdf", "rb") as f:
    resp = requests.put(
        upload_url,
        data=f,
        headers={
            "Content-Type": "application/pdf",
            "x-amz-meta-corpus_id": "f615d1b4-64a5-436e-8baa-2829cae68444",
            "x-amz-meta-original_filename": "arpon-kapuria-cv.pdf",
        },
    )
print(resp.status_code, resp.text)
