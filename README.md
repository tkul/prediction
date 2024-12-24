Öncelikle bir `.venv` klasörü oluşturun.

```bash
python -m venv .venv
```

Virtual enviroment içine girin.

Linux/MacOS:
```bash
source .venv/bin/activate
```
Windows:
```bash
source .venv/Scripts/activate
```

---

Gerekli paketleri indirelim:
```bash
pip install -r requirements.txt
```

---

Kodu çalıştıralım:
```bash
python main.py
```