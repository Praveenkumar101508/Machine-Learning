
from colorama import Fore, Style, init
import os
import matplotlib.pyplot as plt
import joblib
from config import ALLOWED_EXTENSIONS, OUTPUT_GRAPHS, VARIABLES_PATH
import logging

logger = logging.getLogger(__name__)

# Initialize colorama for colored output
init(autoreset=True)

def print_colored(text, color="white", style="normal"):
    color_map = {
        "white": Fore.WHITE,
        "yellow": Fore.YELLOW,
        "green": Fore.GREEN,
        "cyan": Fore.CYAN,
        "red": Fore.RED
    }
    style_map = {
        "normal": Style.NORMAL,
        "bright": Style.BRIGHT
    }
    print(f"{style_map.get(style, Style.NORMAL)}{color_map.get(color, Fore.WHITE)}{text}{Style.RESET_ALL}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_output_dir():
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

def save_plot(fig, filename):
    filepath = os.path.join(OUTPUT_GRAPHS, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved: {filepath}")

def load_variables():
    try:
        best_model, data = joblib.load(VARIABLES_PATH)
        return best_model, data
    except Exception as e:
        logger.error(f"Error loading variables: {str(e)}")
        return None, None