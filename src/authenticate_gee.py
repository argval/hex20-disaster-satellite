import ee

print("Authenticating Google Earth Engine...")
ee.Authenticate()

print("Initializing Google Earth Engine with project ID: hex20-disaster-satellite...")
ee.Initialize(project='hex20-disaster-satellite')

print("Google Earth Engine initialized successfully.")
