import typer
from pathlib import Path
from datetime import datetime
import json
from src import train_agent, quantize, export_esp32, make_test_input

app = typer.Typer(help="PhysioFusion Agent: train -> quantize -> generate test input -> deploy to ESP-IDF project")

@app.command(name="test_pipeline")
def test_pipeline(
    esp_project: str = typer.Option(..., help="Path to ESP-IDF project (folder containing CMakeLists.txt)"),
    csv_path: str = typer.Option(..., help="CSV file containing column A1 to use as test input (>=256 samples)"),
    port: str = typer.Option(None, help="Serial port for flashing, e.g., COM5. If omitted, agent will only build."),
    col: str = typer.Option("A1", help="CSV column name to use (default A1)"),
    n: int = typer.Option(250, help="Number of samples to use for test input"),
):
    """End-to-end local test flow (dummy model + CSV test input)."""
    esp_project = str(Path(esp_project).resolve())
    csv_path = str(Path(csv_path).resolve())

    model_path = train_agent.run_training()
    print("Best model:", model_path)

    tflite_path = quantize.to_int8(model_path)
    print("Quantized model:", tflite_path)

    main_dir = export_esp32.get_main_dir(esp_project)

    model_c = export_esp32.generate_model_c(tflite_path, str(Path(main_dir) / "model_data.c"))
    print("model_data.c written:", model_c)

    test_c = make_test_input.csv_to_test_input_c(csv_path, str(Path(main_dir) / "test_input.c"), col=col, n=n)
    print("test_input.c written:", test_c)

    # Generate inference code
    main_c = export_esp32.generate_inference_main_c(main_dir)
    print("main.c generated:", main_c)

    export_esp32.ensure_main_includes_generated_files(str(Path(main_dir) / "main.c"))

    try:
        export_esp32.idf_build(esp_project)
        if port:
            export_esp32.idf_flash_monitor(esp_project, port)
    except FileNotFoundError:
        print("\n[WARNING] ESP-IDF (idf.py) not found in PATH")
        print("   Files are ready in:", main_dir)
        print("   Next steps:")
        print("   1. Set up ESP-IDF environment")
        print("   2. Run: idf.py build")
        print("   3. Run: idf.py -p COM4 flash monitor")

    print("\nDone. Files copied into ESP-IDF main/.\n"
          "Next step: add TFLite Micro inference code in main.c so firmware actually runs the model.")

@app.command(name="deploy_and_test")
def deploy_and_test(
    esp_project: str = typer.Option(..., help="Path to ESP-IDF project (folder containing CMakeLists.txt)"),
    csv_path: str = typer.Option(..., help="CSV file containing test data (>=256 samples)"),
    port: str = typer.Option(..., help="Serial port for flashing, e.g., COM5"),
    col: str = typer.Option("A1", help="CSV column name to use (default A1)"),
    n: int = typer.Option(256, help="Number of samples to use for test input"),
    output_dir: str = typer.Option("./deployment_results", help="Directory to save deployment results and metrics"),
):
    """Automated workflow: Train -> Quantize -> Deploy -> Test Performance."""
    
    esp_project = str(Path(esp_project).resolve())
    csv_path = str(Path(csv_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    results = {
        "timestamp": timestamp,
        "esp_project": esp_project,
        "csv_path": csv_path,
        "port": port,
        "stages": {}
    }
    
    try:
        # Stage 1: Train
        print("\n" + "="*60)
        print("STAGE 1: Training Model")
        print("="*60)
        best_model_path, metrics = train_agent.run_training()
        print("✓ Best model:", best_model_path)
        print("✓ Metrics:", metrics)
        results["stages"]["training"] = {
            "status": "success",
            "model_path": best_model_path,
            "metrics": metrics
        }
        
        # Stage 2: Quantize
        print("\n" + "="*60)
        print("STAGE 2: Quantizing Model (INT8)")
        print("="*60)
        tflite_path = quantize.to_int8(best_model_path)
        tflite_size = Path(tflite_path).stat().st_size
        print(f"✓ Quantized model: {tflite_path}")
        print(f"✓ Model size: {tflite_size / 1024:.2f} KB")
        results["stages"]["quantization"] = {
            "status": "success",
            "tflite_path": tflite_path,
            "size_bytes": tflite_size
        }
        
        # Stage 3: Deploy
        print("\n" + "="*60)
        print("STAGE 3: Deploying to ESP32")
        print("="*60)
        main_dir = export_esp32.get_main_dir(esp_project)
        
        model_c = export_esp32.generate_model_c(tflite_path, str(Path(main_dir) / "model_data.c"))
        print("✓ model_data.c generated:", model_c)
        
        test_c = make_test_input.csv_to_test_input_c(csv_path, str(Path(main_dir) / "test_input.c"), col=col, n=n)
        print("✓ test_input.c generated:", test_c)
        
        export_esp32.ensure_main_includes_generated_files(str(Path(main_dir) / "main.c"))
        print("✓ main.c includes updated")
        
        export_esp32.idf_build(esp_project)
        print("✓ Firmware built successfully")
        
        export_esp32.idf_flash_monitor(esp_project, port)
        print("✓ Firmware flashed to ESP32")
        
        results["stages"]["deployment"] = {
            "status": "success",
            "model_c": model_c,
            "test_c": test_c,
            "main_dir": main_dir
        }
        
        # Stage 4: Performance Testing
        print("\n" + "="*60)
        print("STAGE 4: Performance Testing")
        print("="*60)
        print("⏳ Waiting for ESP32 inference results...")
        print("📝 Monitor output above for performance metrics")
        print("💡 Expected metrics: latency (ms), inference time, memory usage")
        
        results["stages"]["testing"] = {
            "status": "in_progress",
            "note": "Monitor serial output for real-time metrics"
        }
        
        # Save results
        results_file = Path(output_dir) / f"deployment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
        
        print("\n" + "="*60)
        print("DEPLOYMENT WORKFLOW COMPLETE")
        print("="*60)
        print(f"Model deployed to: {esp_project}")
        print(f"Serial port: {port}")
        print(f"Results: {results_file}")
        print("\nNext steps:")
        print("1. Monitor the ESP32 serial output for inference results")
        print("2. Verify performance metrics (latency, accuracy)")
        print("3. Adjust main.c inference code as needed")
        
    except Exception as e:
        print(f"\n✗ Error during deployment: {str(e)}")
        results["stages"]["error"] = {
            "status": "failed",
            "error": str(e)
        }
        results_file = Path(output_dir) / f"deployment_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        raise

if __name__ == "__main__":
    app()
