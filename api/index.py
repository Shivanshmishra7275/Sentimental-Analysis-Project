from src.app import app  # Re-export the global ASGI app for Vercel

# Vercel's Python runtime will look for a top-level `app` object
# when treating this file as an ASGI application entrypoint.
