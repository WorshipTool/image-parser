# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Requirements

Write all code comments, documentation, and descriptions in English, regardless of the language used in questions or requests. Never create git commits or perform any git write operations - git is read-only.

## Project Overview

Image-Parser is a Python service that extracts Christian hymn/song information from photographs. It detects songs using YOLO8, performs OCR with Tesseract, and structures results into JSON for the Chvalotce.cz web application. Part of the WorshipTool ecosystem.

## Common Commands

### Local Development

```bash
pip install -r requirements.txt      # Install dependencies
python main.py image.jpg             # Parse single image
python main.py image.jpg --ai        # With AI corrections (requires OPENAI_API_KEY)
python main.py image.jpg --debug     # Debug output
python main.py *.jpg -o output/      # Custom output directory
```

### Running the Server (requires Redis)

```bash
python -m server                     # Start Flask server (port 6610)
python -m server.worker              # Start RQ worker (separate terminal)
```

### Docker Deployment

```bash
make setup                           # Create .env from template
make build                           # Build images
make up                              # Start services (server + worker + redis)
make logs                            # View all logs
make rebuild                         # Rebuild and restart
```

### Testing

```bash
pytest                               # Run all tests
pytest parser/text_parser/tests/     # Run specific test module
make test                            # Health check (Docker)
make test-upload                     # Test file upload (Docker)
```

## Architecture

```
CLI (main.py)
    │
    ▼
Parser Module (parser/)
    ├── paper_detection/    → U-Net model for detecting paper edges
    ├── paper_transform/    → Perspective correction
    ├── sheet_detection/    → YOLO8 detects: sheet, title, data regions
    └── text_parser/        → OCR + formatting into JSON
    │
    ▼
Server (server/)
    ├── app.py              → Flask routes + Swagger UI (/docs)
    ├── api.py              → Parser wrapper with progress reporting
    ├── processor/          → RQ job handler
    └── tech/               → Queue & file management

Queue: Redis + RQ (async processing)
Dashboard: /board (RQ Dashboard)
```

### Key Detection Classes (YOLO)

- `sheet` - Entire song page
- `title` - Song title
- `data` - Song body (lyrics, chords)

### Output Format

JSON with `title` field and `data` containing formatted text with inline chords `[Am]` and section markers `{V1}`, `{Chorus1}`.

## Key Files

| File                                 | Purpose                                       |
| ------------------------------------ | --------------------------------------------- |
| `main.py`                            | CLI entry point                               |
| `parser/parse.py`                    | Core CLI implementation                       |
| `parser/get_sheet_components.py`     | Image preprocessing pipeline                  |
| `parser/sheet_detection/__init__.py` | YOLO8 model wrapper                           |
| `parser/text_parser/__init__.py`     | OCR and text parsing (`read_and_parse_image`) |
| `parser/paper_detection/detector.py` | Paper edge detection                          |
| `server/app.py`                      | Flask application & routes                    |
| `server/processor/__init__.py`       | Job processor (`parse_file_func`)             |
| `yolo8best.pt`                       | Pre-trained YOLO8 model                       |

## Environment Variables

```bash
OPENAI_API_KEY=...          # Optional, for AI corrections
PORT=6610                   # Server port
REDIS_HOST=redis            # Redis host (default for Docker)
BRIDGE_URL=...              # Optional, WorshipTool service discovery
```

## API Endpoints

- `GET /is-available` - Health check
- `POST /parse-file?useAi=false` - Synchronous parsing
- `POST /add-file-to-parse-queue` - Async job queue
- `GET /get-job-status-stream?id=ID` - SSE progress stream
- `GET /docs` - Swagger documentation
