from __future__ import annotations

from ipaddress import ip_address

from fastapi import HTTPException, Request


FC_WARMUP_PATH = "/_fc/warmup"


def require_loopback(request: Request) -> None:
    client = request.client
    try:
        is_loopback = client is not None and ip_address(client.host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(status_code=404, detail="Not Found")
