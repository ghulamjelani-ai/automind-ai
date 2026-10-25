# AI Marketing Automation Platform (Mistral 7B + Stable Diffusion 1.5)

An AI marketing assistant for small businesses. Type a short prompt such as "Eid sale on our cakes" and the app answers in seconds with three ready-to-post captions, descriptions and hashtags. Each post comes with a picture drawn by Stable Diffusion. You can publish or schedule posts straight to your Facebook Page.

This edition runs on open models that you fine-tune yourself:

- **Mistral 7B Instruct v0.3**, QLoRA fine-tuned on marketing examples, writes the captions, descriptions, hashtags, plans, weekly posts and posting times.
- **Stable Diffusion 1.5**, LoRA fine-tuned on your own product or brand photos, draws the post images.

## Features

- **Instant posts from any prompt:** every message that names a product, offer or business gets three posts with Caption, Description, Hashtags and Image prompt.
- **Marketing plans:** ask for a plan, strategy or campaign and get a seven-section plan with ready posts.
- **AI images:** Stable Diffusion 1.5 with automatic prompt rewriting for SD and an optional brand-style LoRA.
- **Autopilot:** one click writes a post, draws the picture and publishes or schedules it.
- **7-day auto-poster:** a week of posts, each with an optional picture, scheduled on Facebook.
- **Business memory:** the app remembers lasting facts about your business across chats.
- **Best time to post:** suggested times from your own post performance, or from the AI.
- **Saved chat history, user accounts and Facebook publishing**, the same as the original project.
- **AI engine status badge** in the navbar shows whether the GPU server is ready and fine-tuned.

## Tech Stack

| Layer          | Technology                                                      |
| -------------- | --------------------------------------------------------------- |
| Frontend       | React 19, Vite, Tailwind CSS 4, React Router, Axios             |
| Backend        | FastAPI, Uvicorn, SQLAlchemy                                    |
| Database       | PostgreSQL                                                      |
| Authentication | Firebase Authentication (client) + Firebase Admin SDK (server)  |
| Text AI        | Mistral 7B Instruct v0.3 + LoRA (transformers, peft, bitsandbytes) |
| Image AI       | Stable Diffusion 1.5 + LoRA (diffusers)                         |
| Model hosting  | `model_server/` (FastAPI) on a Google Colab GPU, exposed via a Cloudflare or ngrok tunnel |
| Social         | Facebook Graph API (v23.0)                                      |

## How It Works

```
React frontend ──(Firebase ID token)──▶ FastAPI backend (your PC) ──▶ PostgreSQL
                                              │
                                              ├──(X-API-Key, HTTPS tunnel)──▶ Model server on Colab GPU
                                              │                                 ├─ Mistral 7B + marketing LoRA
                                              │                                 └─ Stable Diffusion 1.5 + brand LoRA
                                              └──▶ Facebook Graph API
```

Your PC has no NVIDIA GPU, so the two models run on a free Colab T4 GPU. The backend, the frontend and the database stay on your PC. The backend sends only the prompt text to the model server.

## Project Structure

```
marketing-ai-local/
├── backend/                  # FastAPI app (same features as the original project)
│   └── app/core/
│       ├── prompts.py        # System prompts + message builders (shared with training)
│       ├── ai_client.py      # Posts, plans, week plans, memory, times, images
│       └── model_server.py   # HTTP client for the GPU server, with clear errors and retries
├── model_server/             # Runs on the GPU: Mistral 7B + SD 1.5 behind an API key
│   ├── server.py             # /health, /v1/chat, /v1/image
│   ├── text_engine.py        # 4-bit or fp16 Mistral + LoRA adapter
│   ├── image_engine.py       # SD 1.5 (DPM++ 2M Karras) + LoRA, safety checker, 2x upscale
│   └── chat_format.py        # Mistral turn format (shared with training)
├── training/
│   ├── data/seed_examples.jsonl   # 47 hand-written examples covering every task
│   ├── data/custom/               # your own extra examples (*.jsonl)
│   ├── data/brand_images/         # your photos for the image LoRA
│   ├── export_from_db.py          # turns your saved chats into training examples
│   ├── build_dataset.py           # renders all examples with the app's real prompts
│   ├── train_mistral_lora.py      # QLoRA fine-tuning of Mistral 7B
│   └── train_sd_lora.py           # LoRA fine-tuning of SD 1.5
├── colab/MarketingAI_GPU.ipynb    # one notebook: install, train, serve, print the URL
└── frontend/                 # React app (unchanged UI, plus the AI engine badge)
```

## Getting Started

### Prerequisites

- Python 3.10+, Node.js 20+, PostgreSQL
- A Firebase project (Email/Password and Google sign-in)
- A Google account for Colab and Google Drive
- A [Hugging Face](https://huggingface.co/settings/tokens) token with read access. Accept the terms on the [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) page first, because the model is gated.
- A Meta app with Facebook Login (only needed for publishing)

### 1. Database

The database is the same as in the original project, and the tables are created automatically on startup.

```sql
CREATE DATABASE marketing_ai_db;
CREATE USER marketing_ai_user WITH PASSWORD 'choose-a-strong-password';
GRANT ALL PRIVILEGES ON DATABASE marketing_ai_db TO marketing_ai_user;
```

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and fill it in. Put your Firebase service account file at `backend/firebase-service-account.json`.

### 3. Optional: use your existing chats as training data

Your saved chats already follow the post and plan format. To export them, run this from the project folder with the backend venv active:

```bash
python training/export_from_db.py
```

This writes `training/data/db_export.jsonl`. Review it before uploading.

### 4. GPU model server on Colab (train + serve)

1. Upload the whole `marketing-ai-local` folder to Google Drive as `MyDrive/marketing-ai-local`. You can leave out `frontend/node_modules` and `backend/venv`.
2. Optional: put 15 to 60 brand or product photos in `training/data/brand_images/`.
3. Open `colab/MarketingAI_GPU.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
4. Add the Colab secret `HF_TOKEN` (key icon). `NGROK_AUTHTOKEN` is optional.
5. Select **Runtime → Run all**. The first run fine-tunes Mistral (about 15 to 40 minutes) and SD 1.5 if you added photos (about 10 to 20 minutes). The adapters are saved on Drive, so later runs skip training and start in about 5 minutes.
6. The last cell prints:

```
MODEL_SERVER_URL=https://xxxx.trycloudflare.com
MODEL_SERVER_API_KEY=...
```

Paste both lines into `backend/.env`. The key stays the same between sessions. With Cloudflare, the URL changes each session.

### 5. Run the app

On Windows, double-click `start.bat`. It opens the backend and frontend in two windows and then opens the browser. The first run also installs anything that is missing.

Or start them by hand:

```bash
cd backend
uvicorn app.main:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The navbar badge shows **AI engine ready** when the GPU server is reachable.

## Training details

| | Mistral 7B | Stable Diffusion 1.5 |
|---|---|---|
| Method | QLoRA (4-bit NF4 base, LoRA r=16 on all attention and MLP layers) | LoRA r=16 on UNet attention |
| Data | seed examples + your chats + `custom/*.jsonl` | your photos + optional captions |
| Default | 3 epochs, lr 2e-4, cosine schedule | about 80 steps per photo (400 to 1500), lr 1e-4 |
| Output | `training/output/mistral-marketing-lora` | `training/output/sd15-brand-lora` (+ `trigger.txt`) |

The training examples are rendered with the same prompt builders as the backend (`backend/app/core/prompts.py`) and the same Mistral turn format (`model_server/chat_format.py`). The fine-tuned model sees exactly the prompts it will get in the app. To retrain after adding examples or photos, set `RETRAIN_TEXT` or `RETRAIN_IMAGE = True` in the notebook.

## Speed (Colab T4)

| Task | Time |
| --- | --- |
| 3 posts (caption, description, hashtags, image prompt) | about 20 to 35 s |
| Full 7-section plan | about 45 to 70 s |
| One image (512 px, 28 steps, upscaled to 1024 px) | about 5 to 8 s, plus a few seconds for the prompt rewrite |
| 7-day plan | about 60 to 90 s (two batches) |

A bigger GPU, such as an L4 or A100 on Colab Pro, runs Mistral in fp16 automatically and is 2 to 3 times faster.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Health check |
| GET | `/ai/status` | GPU model server status (models, fine-tuned or not, GPU) |
| GET | `/auth/me`, POST `/auth/sync` | User profile |
| POST | `/chat/plan` | Chat: posts, plans and follow-ups (Mistral) |
| GET | `/chat/conversations`, `/chat/history/{id}` | Chat history |
| POST | `/images/generate` | Generate a PNG image (Stable Diffusion 1.5) |
| POST | `/autopilot/post` | One ready-to-publish post |
| POST | `/schedule/draft` | A week of posts |
| GET | `/schedule/best-time` | Suggested posting times |
| GET/PUT/DELETE | `/memory` | Business memory |
| | `/facebook/*`, `/schedule/*`, `/history/*` | Facebook connection, publishing, scheduling, history |

Model server endpoints (all except `/health` require the `X-API-Key` header):

| Method | Endpoint | Body |
| --- | --- | --- |
| GET | `/health` | Model loading status |
| POST | `/v1/chat` | `{messages, max_new_tokens, temperature, top_p, prefix}` → `{text, tokens, seconds}` |
| POST | `/v1/image` | `{prompt, negative_prompt, steps, guidance_scale, width, height, seed}` → `{image_base64, seed, seconds}` |

## Troubleshooting

| Message in the app | Fix |
| --- | --- |
| "The AI model server is not configured" | Set `MODEL_SERVER_URL` and `MODEL_SERVER_API_KEY` in `backend/.env` and restart the backend. |
| "Could not reach the AI model server" / "offline" | The Colab session stopped or the tunnel URL changed. Run the notebook again and update `MODEL_SERVER_URL`. |
| "rejected the API key" | Copy the key from the notebook's last cell into `backend/.env` again. |
| "still loading on the GPU" | Wait until the notebook's server cell shows `ready`. |
| "blocked by the safety filter" | Reword the image description. |
| Notebook: `401` or `gated repo` when loading Mistral | Accept the model terms on Hugging Face and check the `HF_TOKEN` secret. |

## Security

Never commit `backend/.env`, `firebase-service-account.json`, `model_server/.env` or `model_server/.api_key`. They are listed in `.gitignore`. The model server is public through the tunnel, so every model call requires the API key.
#   a u t o m i n d - a i  
 