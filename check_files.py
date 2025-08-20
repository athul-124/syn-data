import os
import glob

# Check generated directory
print("Generated files:")
for file in glob.glob("generated/*.csv"):
    print(f"  {file}")

print("\nGenerated reports:")
for file in glob.glob("generated/reports/*.json"):
    print(f"  {file}")