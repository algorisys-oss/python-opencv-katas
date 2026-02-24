
try:
    # This will raise SyntaxError during parsing, so we need to eval it to catch it in a running script
    # or just demonstrate the string parsing.
    # Actually, SyntaxError happens at parse time, so we can't try-except it easily in the same block 
    # unless we use exec/eval.
    exec('path = "C:\\Users\\Vishaka\\Downloads\\pikachu.jfif"')
    print("SUCCESS: Normal string worked (unexpected)")
except SyntaxError as e:
    print(f"CAUGHT EXPECTED ERROR: {e}")

print("-" * 20)

try:
    # Fix 1: Raw string
    exec('path = r"C:\\Users\\Vishaka\\Downloads\\pikachu.jfif"')
    print("SUCCESS: Raw string worked")
except Exception as e:
    print(f"FAILED: Raw string: {e}")

try:
    # Fix 2: Double backslash
    exec('path = "C:\\\\Users\\\\Vishaka\\\\Downloads\\\\pikachu.jfif"')
    print("SUCCESS: Double backslash worked")
except Exception as e:
    print(f"FAILED: Double backslash: {e}")

try:
    # Fix 3: Forward slash
    exec('path = "C:/Users/Vishaka/Downloads/pikachu.jfif"')
    print("SUCCESS: Forward slash worked")
except Exception as e:
    print(f"FAILED: Forward slash: {e}")
