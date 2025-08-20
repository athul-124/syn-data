import os

# Create necessary directories
os.makedirs("generated", exist_ok=True)
os.makedirs("generated/reports", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("✅ Created directories: generated/, generated/reports/, uploads/, outputs/")