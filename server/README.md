# Image Parser Server

Flask-based REST API server for asynchronous image parsing with Redis Queue (RQ) support.

## Overview

This server provides HTTP endpoints for parsing Christian hymn/song sheet images. It supports both synchronous (blocking) and asynchronous (queue-based) processing with real-time progress updates via Server-Sent Events (SSE).

## Architecture

```
Client (HTTP)
    ↓
Flask Server (server.py)
    ├─ Swagger UI (/docs)
    ├─ RQ Dashboard (/board)
    └─ API Endpoints
    ↓
Redis Queue (RQ)
    ├─ Queue: "parse_files"
    └─ Job metadata (progress %)
    ↓
RQ Worker (run_worker.sh)
    └─ processor(files, useAi)
    ↓
Parser API (server/api.py)
    └─ parse_images(paths, use_ai, debug)
    ↓
Parser Module (../parser/)
    ├─ YOLO Detection
    ├─ OCR Text Reading
    ├─ Text Formatting
    └─ Optional AI Corrections
    ↓
JSON Results
```

## Components

### Directory Structure

```
server/
├── README.md                  # This file
├── __main__.py                # Module entry point (python -m server)
├── app.py                     # Main Flask application
├── api.py                     # Parser API wrapper with progress reporting
├── processor/                 # Job processing orchestration
│   └── __init__.py           # RQ job handler and file processor
├── tech/                      # Technical utilities
│   └── __init__.py           # Queue management and file handling
├── constants/                 # Configuration
│   └── __init__.py           # Queue name and temp folder paths
└── swagger/                   # API documentation
    ├── is-available.yml      # Health check endpoint spec
    ├── parse-file.yml        # Sync processing endpoint spec
    └── get-job-status.yml    # Async status endpoint spec

../main.py                     # CLI wrapper for parser
```

### Core Files

**`app.py`** - Main Flask application
- API endpoint routing
- Swagger UI integration (`/docs`)
- RQ Dashboard integration (`/board`)

**`api.py`** - Parser wrapper API
- `parse_images(paths, use_ai, debug)` - Batch image processing with progress
- `parse_single_image(path, use_ai, debug)` - Single image convenience wrapper
- Generator-based progress reporting (yields 0-100%)

**`processor/__init__.py`** - Job processing
- `parse_file_func()` - Core processing logic with file cleanup
- `processor()` - RQ job wrapper with progress tracking
- Handles exceptions and temporary file deletion

**`tech/__init__.py`** - Queue & file utilities
- `add_to_queue(files, useAi)` - Enqueue files for processing
- `get_job(job_id)` - Retrieve job status from Redis
- `save_files(files)` - Save uploaded files to temp directory
- `generate_filename()` - UUID-based filename generation

**`constants/__init__.py`** - Configuration
- `QUEUE_NAME = "parse_files"` - Redis queue name
- `TEMP_FOLDER = "tmp"` - Temporary file storage path

## API Endpoints

### Health Check

```http
GET /is-available
```

**Response:**
```json
{
  "isAvailable": true
}
```

### Synchronous Processing (Blocking)

```http
POST /parse-file?useAi=false
Content-Type: multipart/form-data

file: <image.jpg>
```

**Parameters:**
- `useAi` (query, optional): Enable AI corrections (default: `false`)

**Response:** (200 OK)
```json
[
  {
    "title": "Amazing Grace",
    "data": "{V1}[Am]Amazing grace...\n{Chorus1}[G]How sweet...",
    "inputImagePath": "sheet.jpg"
  }
]
```

### Asynchronous Processing - Submit

```http
POST /add-file-to-parse-queue?useAi=false
Content-Type: multipart/form-data

file: <image.jpg>
```

**Response:** (200 OK)
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Asynchronous Processing - Stream Status

```http
GET /get-job-status-stream?id=550e8400-e29b-41d4-a716-446655440000
```

**Response:** (200 OK, `text/event-stream`)
```
event: progress
data: {"status": "queued", "progress": 0}

event: progress
data: {"status": "started", "progress": 15}

event: progress
data: {"status": "started", "progress": 45}

event: final
data: {"status": "finished", "results": [...]}
```

### Asynchronous Processing - Poll Status

```http
GET /get-job-result?id=550e8400-e29b-41d4-a716-446655440000
```

**Response:** (202 Accepted) - Job still processing
```json
{
  "status": "started",
  "progress": 45
}
```

**Response:** (200 OK) - Job completed
```json
{
  "status": "finished",
  "results": [
    {
      "title": "Amazing Grace",
      "data": "...",
      "inputImagePath": "sheet.jpg"
    }
  ]
}
```

## Configuration

### Environment Variables (`.env`)

```bash
# OpenAI API (required if useAi=true)
OPENAI_API_KEY=sk-...

# Server Configuration
PORT=6610
HOST=0.0.0.0

# Redis Configuration (default)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

### Application Settings

```python
# server.py
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

# constants/__init__.py
QUEUE_NAME = "parse_files"
TEMP_FOLDER = "tmp"
```

## Running the Server

### Prerequisites

1. **Redis Server** - Must be running
```bash
redis-server
```

2. **Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
# Create .env file with required variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Start Server

**Terminal 1 - Flask Server:**
```bash
# From image-parser root directory (recommended)
python -m server

# Or with custom port
PORT=8000 python -m server
```

Output:
```
Starting server on 0.0.0.0:6610
  - API: http://0.0.0.0:6610
  - Docs: http://0.0.0.0:6610/docs
  - Dashboard: http://0.0.0.0:6610/board
 * Running on http://0.0.0.0:6610
```

**Terminal 2 - RQ Worker:**
```bash
bash server/run_worker.sh
```

Output:
```
Worker rq:worker:... started
Listening on queue: parse_files
```

### Access Points

- **API Server**: http://localhost:6610
- **Swagger Documentation**: http://localhost:6610/docs
- **RQ Dashboard**: http://localhost:6610/board
- **Health Check**: http://localhost:6610/is-available

## Usage Examples

### cURL - Synchronous

```bash
curl -X POST http://localhost:6610/parse-file?useAi=false \
  -F "file=@sheet.jpg"
```

### cURL - Asynchronous

```bash
# Submit job
JOB_ID=$(curl -X POST http://localhost:6610/add-file-to-parse-queue?useAi=true \
  -F "file=@sheet.jpg" | jq -r '.id')

# Stream progress
curl http://localhost:6610/get-job-status-stream?id=$JOB_ID

# Or poll for result
curl http://localhost:6610/get-job-result?id=$JOB_ID
```

### Python Client

```python
import requests

# Synchronous
with open('sheet.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:6610/parse-file?useAi=false',
        files={'file': f}
    )
    results = response.json()
    print(results[0]['title'])

# Asynchronous
with open('sheet.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:6610/add-file-to-parse-queue?useAi=true',
        files={'file': f}
    )
    job_id = response.json()['id']

# Stream progress
import sseclient
response = requests.get(
    f'http://localhost:6610/get-job-status-stream?id={job_id}',
    stream=True
)
client = sseclient.SSEClient(response)
for event in client.events():
    print(f"{event.event}: {event.data}")
```

### JavaScript Client

```javascript
// Asynchronous with SSE
const formData = new FormData();
formData.append('file', fileInput.files[0]);

// Submit job
const submitResponse = await fetch(
  'http://localhost:6610/add-file-to-parse-queue?useAi=true',
  { method: 'POST', body: formData }
);
const { id } = await submitResponse.json();

// Stream progress
const eventSource = new EventSource(
  `http://localhost:6610/get-job-status-stream?id=${id}`
);

eventSource.addEventListener('progress', (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}%`);
});

eventSource.addEventListener('final', (event) => {
  const data = JSON.parse(event.data);
  console.log('Results:', data.results);
  eventSource.close();
});
```

## Processing Flow

### With AI Enabled (`useAi=true`)

1. **Sheet Detection** (0-30%) - YOLO model detects sheet regions
2. **OCR Processing** (30-60%) - Extract text from detected regions
3. **AI Line Correction** (60-75%) - Fix OCR errors per line
4. **AI Final Correction** (75-95%) - Validate and correct full sheet
5. **Formatting** (95-100%) - Structure output as JSON

### Without AI (`useAi=false`)

1. **Sheet Detection** (0-40%) - YOLO model detects sheet regions
2. **OCR Processing** (40-80%) - Extract text from detected regions
3. **Formatting** (80-100%) - Structure output as JSON

## Job States

- `queued` - Job waiting in Redis queue
- `started` - Worker processing job
- `finished` - Job completed successfully
- `failed` - Job encountered error

## Monitoring

### RQ Dashboard

Access at http://localhost:6610/board

Features:
- View queued jobs
- Monitor worker status
- Inspect job details
- Retry failed jobs
- Clear queues

### Logs

Server logs show:
```
Processing: sheet.jpg
✓ Detected 2 sheets
✓ Parsed: "Amazing Grace"
✓ Parsed: "How Great Thou Art"
```

Worker logs show:
```
parse_files: processor(files=[...], useAi=True) (job_id)
parse_files: Job OK (1.2s)
```

## Error Handling

### Server Errors

- **400 Bad Request** - Invalid file format or missing file
- **500 Internal Server Error** - Processing exception

### Job Errors

Jobs that fail return:
```json
{
  "message": "Error description"
}
```

Uploaded files are automatically cleaned up even on error.

## Performance

### Benchmarks (approximate)

- **Without AI**: ~2-5 seconds per sheet
- **With AI**: ~10-20 seconds per sheet (depends on OpenAI API latency)

### Optimization Tips

1. Use `useAi=false` for faster processing when OCR quality is good
2. Process multiple images in parallel by submitting multiple jobs
3. Use async endpoints for better UX with long-running jobs
4. Scale workers horizontally: `rq worker parse_files --name worker2`

## Troubleshooting

### Redis Connection Failed

```bash
# Check Redis is running
redis-cli ping  # Should return "PONG"

# Start Redis if not running
redis-server
```

### Worker Not Processing Jobs

```bash
# Check worker is running
rq info --url redis://localhost:6379

# Restart worker
bash server/run_worker.sh
```

### OpenAI API Errors (when useAi=true)

```bash
# Check API key is set
echo $OPENAI_API_KEY

# Verify .env file
cat .env | grep OPENAI_API_KEY
```

### Port Already in Use

```bash
# Find process using port 6610
lsof -i :6610

# Kill process
kill -9 <PID>

# Or use different port
PORT=6611 python server.py
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-flask

# Run tests
pytest server/tests/
```

### Adding New Endpoints

1. Add route in `server.py`
2. Create Swagger spec in `server/swagger/`
3. Update this README with endpoint documentation

## Related Documentation

- **Parser Module**: `../parser/README.md`
- **Main Parser CLI**: `../parser/parse.py --help`
- **API Reference**: http://localhost:6610/docs (when server is running)
