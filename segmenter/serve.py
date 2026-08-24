from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "9000"))
    keep_alive_timeout = int(os.getenv("KEEP_ALIVE_TIMEOUT", "900"))
    uvicorn.run(
        "segmenter.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=keep_alive_timeout,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()

