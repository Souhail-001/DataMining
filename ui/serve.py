"""Lance l'interface en local : python ui/serve.py  (puis le navigateur s'ouvre).

Le navigateur refuse de lire `data/` quand on ouvre index.html directement
(file://) : il faut un petit serveur HTTP. Celui de la bibliotheque standard suffit.
Si `ui/data/` n'existe pas encore : python scripts/export_ui.py
"""

import functools
import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

if not (HERE / "data" / "meta.json").is_file():
    sys.exit("ui/data/meta.json manquant : lancez d'abord  python scripts/export_ui.py")

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
Handler.log_message = lambda *args: None  # console silencieuse

with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler) as httpd:
    url = f"http://127.0.0.1:{PORT}/index.html"
    print(f"Interface disponible sur {url}  (Ctrl+C pour arreter)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArret.")
