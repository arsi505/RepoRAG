# RepoRAG frontend

Local Next.js interface for RepoRAG's FastAPI repository Q&A service.

## Run locally

Start the API from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn reporag.api.app:app --reload --host 127.0.0.1 --port 8000
```

Then start this frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend defaults to `http://127.0.0.1:8000`; copy `.env.example` to `.env.local` to override that URL.

Provider keys remain in the repository root's ignored `.env`. Never place a provider key in a `NEXT_PUBLIC_*` variable.

## Validate

```powershell
npm test
npm run lint
npm run build
```
