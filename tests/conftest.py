import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def default_judge(prompt):
    """Scores by how many times the hypothesis says 'good'; dialect prefers 'yalla'."""
    hyp = prompt.rsplit("translation: ", 1)[-1].split("\n")[0]
    if "authentically" in prompt:
        return f"Fine.\nScore: {100 if 'yalla' in hyp else 10}"
    return f"**Score:** {min(100, 20 + 40 * hyp.count('good'))}"


class FakeServer:
    """A real HTTP server speaking the OpenAI chat API, to test the whole client path."""

    def __init__(self, reply=default_judge):
        self.reply = reply
        self.requests = []
        self.auth = []
        self.fail_with = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"object": "list", "data": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.auth.append(self.headers.get("Authorization"))
                if outer.fail_with:
                    return self._send(outer.fail_with, {"error": {"message": "nope"}})
                answer = outer.reply(body["messages"][0]["content"])
                self._send(200, {
                    "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": answer}}],
                })

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def judge_server():
    server = FakeServer()
    yield server
    server.close()


@pytest.fixture
def corpus(tmp_path):
    """Two sources with two candidates and one reference each."""
    (tmp_path / "src.txt").write_text("source one\nsource two\n", encoding="utf-8")
    (tmp_path / "hyps.txt").write_text(
        "this is a bad translation of it\nthis is a poor translation here\n"
        "another bad result for you\nanother poor result for us\n", encoding="utf-8")
    (tmp_path / "ref.txt").write_text(
        "this is a fine translation of it\nanother fine result for you\n", encoding="utf-8")
    (tmp_path / "words.txt").write_text("good;yalla;great thing", encoding="utf-8")
    return tmp_path
