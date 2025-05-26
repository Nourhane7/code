from flask import Flask
import os
import signal

def run_flask_app():
    """
    Run Flask application with proper startup handling
    """
    try:
        # Check if the model file exists
        if not os.path.exists('random_forest_model.joblib'):
            raise FileNotFoundError("Model file 'random_forest_model.joblib' not found!")
            
        # Create and run the Flask app
        from app import create_flask_app
        app = create_flask_app()
        
        # Set host to localhost only for security
        app.run(host='127.0.0.1', port=5000, debug=True)
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        # Handle Ctrl+C gracefully
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        
if __name__ == '__main__':
    run_flask_app()
