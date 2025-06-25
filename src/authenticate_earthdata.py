import earthaccess
import getpass

print("NASA Earthdata authentication. Please enter your credentials when prompted.")

try:
    earthaccess.login()
    print("NASA Earthdata authentication successful.")
except Exception as e:
    print(f"NASA Earthdata authentication failed: {e}")