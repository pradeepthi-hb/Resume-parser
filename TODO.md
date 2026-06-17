# TODO

## Objective
Find why hireyo integration shows no AI response even when AI is enabled.

## Planned diagnostics (code)
- [ ] Add server-side debug logging to AI decision points and Gemini call/fallback.
- [ ] Expose fallback/parse mode info in JSON responses from `/detect-headings`, `/api/parse`, `/api/parse-all`.

## Validate
- [ ] Run Flask and call `GET /status`.
- [ ] Trigger `POST /detect-headings` with a sample resume and inspect logs/JSON.
- [ ] Trigger `POST /api/parse-all` and inspect logs/JSON.


