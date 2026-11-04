from evaluate import predict_single_image

# Path to a single test image you want to diagnose
image_path = r"C:\Users\Public\tbxnet_chest_x_ray\dataset\test\TURBERCULOSIS\Tuberculosis-675.png"

# Run prediction
disease_class, confidence = predict_single_image(image_path)

print(f"Prediction Result: {disease_class}")
print(f"Confidence Score: {confidence:.2f}%")