# Your own training examples

Any `.jsonl` file in this folder is added to the Mistral training set by `training/build_dataset.py`.
Write one JSON object per line, in the same shape as `../seed_examples.jsonl`:

```json
{"task": "chat", "message": "eid sale on shoes", "business_name": "Step Up", "output": "Here are three posts...\n\nPost 1\nCaption: ...\nDescription: ...\nHashtags: ...\nImage prompt: ..."}
{"task": "single_post", "request": "weekend offer", "memory": "Step Up sells shoes in Lahore.", "output": "Caption: ...\nDescription: ...\nHashtags: ...\nImage prompt: ..."}
{"task": "week", "count": 4, "description": "...", "output": "Post 1\nCaption: ...\n..."}
{"task": "times", "memory": "...", "output": "1. 09:00 - ...\n2. 13:00 - ...\n3. 20:00 - ..."}
{"task": "memory", "current": "...", "message": "...", "output": "..."}
{"task": "enhance", "idea": "...", "output": "comma, separated, stable diffusion prompt"}
```

Good examples teach the model your tone, your hashtags and your local market. 50 to 300 extra examples make a clear difference.
