import logging
import os

def setup_logging():

    log_dir = "logs"
    os.makedirs(log_dir,exist_ok=True)
    file_name = "Pipeline.log"
    full_path = os.path.join(log_dir,file_name)
    logging.basicConfig(
        level=logging.INFO,
        format= "%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(full_path),logging.StreamHandler()],
    )

    logging.info("Logging initialize successfully")

def main():

    setup_logging()




if __name__ == "__main__":
    main()