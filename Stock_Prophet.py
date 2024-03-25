import os
import subprocess
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import requests
from colorama import init, Fore, Back, Style
from LSTM_Model import process_prediction
from pyfiglet import Figlet
from time import sleep

# ----------------------------------------------------------------------------------------------------------------------
# VARIABLES FOR HEADINGS & MENU OPTIONS
# ----------------------------------------------------------------------------------------------------------------------

# Screen headings
stock_prophet = """
 _____ _             _      ______                _          _   
/  ___| |           | |     | ___ \              | |        | |  
\ `--.| |_ ___   ___| | __  | |_/ / __ ___  _ __ | |__   ___| |_ 
 `--. \ __/ _ \ / __| |/ /  |  __/ '__/ _ \| '_ \| '_ \ / _ \ __|
/\__/ / || (_) | (__|   <   | |  | | | (_) | |_) | | | |  __/ |_ 
\____/ \__\___/ \___|_|\_\  \_|  |_|  \___/| .__/|_| |_|\___|\__|
                                           | |                   
                                           |_|                   
"""
start_h = """
                    ____ ___ ____ ____ ___ 
                    [__   |  |__| |__/  |  
                    ___]  |  |  | |  \  |      
"""
signup_h = """
                  ____ _ ____ _  _    _  _ ___  
                  [__  | | __ |\ |    |  | |__] 
                  ___] | |__] | \|    |__| |        
"""
login_h = """
                    _    ____ ____ _ _  _ 
                    |    |  | | __ | |\ | 
                    |___ |__| |__] | | \|      
"""
exit_h = """
____ ____    _    ____ _  _ ____          
[__  |  |    |    |  | |\ | | __          
___] |__|    |___ |__| | \| |__] .   .   .
"""
home_h = """
_ _ _ ____ ___ ____ _  _ _    _ ____ ___ 
| | | |__|  |  |    |__| |    | [__   |  
|_|_| |  |  |  |___ |  | |___ | ___]  |                                    
"""
search_h = """
    ____ ___ ____ ____ _  _    ____ ____ ____ ____ ____ _  _ 
    [__   |  |  | |    |_/     [__  |___ |__| |__/ |    |__| 
    ___]  |  |__| |___ | \_    ___] |___ |  | |  \ |___ |  |   
"""
commands_h = """
____ ____ _  _ _  _ ____ _  _ ___  ____ 
|    |  | |\/| |\/| |__| |\ | |  \ [__  
|___ |__| |  | |  | |  | | \| |__/ ___] 
                                        
"""
stock_h = Figlet(font='big')

# Menu options
start_m = Back.MAGENTA + "1. Start    " + Style.RESET_ALL + "|    2. Sign Up    |    3. Login    |    4. Exit App"
home_m = Back.MAGENTA + "1. Home    " + Style.RESET_ALL + "|    2. Stocks    |    3. Guide    |    4. Logout"
stocks_m = "1. Home    |" + Back.MAGENTA + "    2. Stocks    " + Style.RESET_ALL + "|    3. Guide    |    4. Logout"
signup_m = "1. Start    |" + Back.MAGENTA + "    2. Sign Up    " + Style.RESET_ALL + "|    3. Login    |    4. Exit App"
login_m = "1. Start    |    2. Sign Up    |" + Back.MAGENTA + "    3. Login    " + Style.RESET_ALL + "|    4. Exit App"


# ----------------------------------------------------------------------------------------------------------------------
# START SCREEN FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def start_screen():
    """A function that displays the Start screen, and awaits user input to
    initiate menu navigation."""

    # Display the 'Start' screen.
    while True:
        screen_clear()
        print(Fore.MAGENTA + Style.BRIGHT + stock_prophet + Style.RESET_ALL)
        print(start_m)
        print()
        print(start_h)
        print()
        print(f"Welcome to Stock Prophet!")
        print()
        print("Stock Prophet is your go-to app for stock price predictions generated")
        print("by machine learning models. Stock Prophet does not provide financial")
        print("advice. This platform is for educational purposes only.")
        print()
        print(Fore.MAGENTA + "HOW IT'S BUILT" + Style.RESET_ALL)
        print("Python: The programming language used for building the application.")
        print("Matplotlib: Library for creating static visualizations.")
        print("YFinance: Library for fetching historical market data from Yahoo Finance.")
        print("Colorama: Library for adding color and style to terminal output.")
        print("Pyfiglet: Library for creating ASCII text banners.")
        print()
        print(Fore.MAGENTA + "KEY FEATURES" + Style.RESET_ALL)
        print("User Authentication: Login and signup functionality.")
        print("Stock Search: Search and view detailed information about stocks.")
        print("Watchlist: Add and remove stocks from your watchlist for easy tracking.")
        print("Guide: Access detailed walkthroughs, FAQs, and additional information.")
        print("Interactive Charts: View historical stock data with static charts.")
        print("Price Predictions: Coming soon! Get predictions for future stock prices.")
        print()
        print(Fore.MAGENTA + "GETTING STARTED" + Style.RESET_ALL)
        print("1) If your are new, enter '2' to choose 'Sign Up'.")
        print("2) Read walkthrough on the 'Sign Up' screen (est. time ~ 1 minute).")
        print("3) Complete membership application (est. time ~ 1 minute).")
        print("4) Read walkthrough on the 'Home' screen (est. time ~ 3 minutes).")
        print(Fore.MAGENTA + "                        OR" + Style.RESET_ALL)
        print("1) If you are a returning member, enter '3' to choose 'Login'.")
        print("2) Enter login credentials")
        print("3) Open 'Guide' for further assistance.")
        print()
        print()
        print("Note: You may enter a menu number at any time.")
        print()

        # Ask user to enter a menu option.
        choice = input("Enter a menu option: ")
        menu_options_1(choice, False)


# ----------------------------------------------------------------------------------------------------------------------
# SIGN UP SCREEN FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def signup_screen():
    """A function that displays the 'Sign Up' screen and calls the enter_credentials
    function to initiate account creation or menu navigation."""

    # Display the 'Sign Up' screen.
    while True:
        screen_clear()
        print(Fore.MAGENTA + Style.BRIGHT + stock_prophet + Style.RESET_ALL)
        print(signup_m)
        print()
        print(signup_h)
        print()
        print()
        print("WALKTHROUGH: CREATE YOUR ACCOUNT")
        print()
        print("- Already have an account? Enter '3' for login page.")
        print()
        print("1) Begin by entering a username (at least 2 characters in length).")
        print("2) Enter a password (any length is appropriate).")
        print("3) Type 'Y' to confirm your credentials or 'N' to restart.")
        print("4) Write down your credentials. We do not have account recovery services.")
        print("5) Follow the walkthrough on the Login screen.")
        print()
        print()
        print("Note: You may enter a menu number at any time.")
        print()

        # Ask user to enter and confirm login credentials.
        create_credentials()


def create_credentials():
    """A function that prompts users to create login credentials."""

    # Ask user to choose username and password, checking if a menu option was chosen.
    username = input("Enter username: ")
    menu_options_1(username, True)

    # Check username for uniqueness.
    username_check(username)

    password = input("Enter password: ")
    menu_options_1(password, True)

    # Ask user to confirm credentials, and then store them in .txt file.
    print()
    print("Are you sure you want this username/password?")
    choice = input("Type 'Y' to confirm or 'N' to restart: ")
    confirm_credentials(username, password, choice)
    save_credentials(username, password)


def username_check(username):
    """A function that determines whether a username already exists."""

    with open("login_credentials.txt", "r") as file:
        for line in file:
            existing_username = line.strip().split(",")
            if existing_username[0] == username:
                print()
                print("Username is already taken. Please choose another one.")
                sleep(3)
                screen_clear()
                signup_screen()


def confirm_credentials(username, password, choice):
    """A function that prompts users to confirm their choice of login credentials,
    and calls the save_credentials function if the user enters 'Y'."""

    if choice == "N" or choice == 'n':
        screen_clear()
        signup_screen()
    elif choice == "Y" or choice == 'y':
        save_credentials(username, password)
    else:
        print()
        print("Error: You did not enter a 'Y' or 'N'. Please try again.")
        sleep(3)
        screen_clear()
        signup_screen()


def save_credentials(username, password):
    """A function that saves the user's newly created login credentials."""

    with open("login_credentials.txt", "a") as file:
        file.write(f"{username},{password}\n")
        print(".")
        print(".")
        print(".")
        print(".")
        print("Sign up successful! Redirecting to Login screen!")
        sleep(3)
    screen_clear()
    login_screen()


# ----------------------------------------------------------------------------------------------------------------------
# LOGIN SCREEN FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def login_screen():
    """A function that displays the 'Login' screen and calls the enter_credentials
    function to initiate login verification or menu navigation."""

    # Display the 'Login' screen.
    while True:
        screen_clear()
        print(Fore.MAGENTA + Style.BRIGHT + stock_prophet + Style.RESET_ALL)
        print(login_m)
        print()
        print(login_h)
        print()
        print()
        print("WALKTHROUGH: LOG INTO YOUR ACCOUNT")
        print()
        print("- Don't have an account? Enter '2' for sign up page.")
        print("- You may have written your login credentials down somewhere.")
        print()
        print("1) Begin by entering your username.")
        print("2) Enter your password.")
        print("3) If you forgot your username and/or password, make a new account.")
        print("4) If login is successful, you will be redirected to the Home screen.")
        print("5) Follow the walkthrough on the Home screen.")
        print()
        print()
        print("Note: You may enter a menu number at any time.")
        print()

        # Ask user to enter login credentials.
        enter_credentials()


def enter_credentials():
    """A function that prompts the user to enter their login credentials or
    enter a menu option."""

    # Ask user to enter their username and password, checking if a menu option was entered.
    username = input("Enter your username: ")
    menu_options_1(username, True)
    password = input("Enter your password: ")
    menu_options_1(password, True)

    # Check if the username and password combination exists.
    check_credentials(username, password)


def check_credentials(username, password):
    """A function that checks whether the entered login credentials exist, and
    logs the user in if said login credentials are found."""

    # Get dictionary of all existing login credentials.
    credentials = get_credentials_dictionary()

    # Redirect to Home screen if username-password combination found in dictionary.
    if username in credentials and credentials[username] == password:
        print(".")
        print(".")
        print(".")
        print(".")
        print("Login successful!")
        sleep(3)
        screen_clear()
        home_screen(username)
    else:
        print()
        print("Invalid username or password. Please try again.")
        sleep(2)
        screen_clear()
        login_screen()


def get_credentials_dictionary():
    """A function that opens and reads login_credentials.txt, and then adds
    each login credentials combination to a dictionary."""
    # Open the login_credentials.txt file in read mode.
    with open("login_credentials.txt", "r") as file:
        lines = file.readlines()

    # Create a dictionary to store username-password pairs.
    credentials = {}
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            username, password = parts
            credentials[username] = password

    # Return the dictionary of login credentials.
    return credentials


# ----------------------------------------------------------------------------------------------------------------------
# HOME SCREEN FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def home_screen(username):
    """A function that displays the 'Home' screen and the user's watchlist,
    and awaits user input to initiate menu navigation."""

    # Initialize user's watchlist.
    watchlist = get_watchlist(username)

    # Display the 'Home' screen and the user's watchlist stock prices.
    while True:
        screen_clear()
        print(Fore.MAGENTA + Style.BRIGHT + stock_prophet + Style.RESET_ALL)
        print(home_m)
        print()
        print()
        print(f"Welcome back, {username}!")
        print()
        print("WALKTHROUGH: MENU OPTIONS & WATCHLIST (See Guide for more details)")
        print()
        print("1) Choosing 'Home' will simply refresh the page and stock prices.")
        print("2) Choosing 'Stocks' will bring you to the 'Stock Search' screen.")
        print("3) Choosing 'Guide' will open a separate file on your pc.")
        print("4) Choosing 'Logout' will log you out and redirect you to the 'Start' screen.")
        print("5) You may visit a Watchlist stock by entering its corresponding ticker.")
        print()
        print(home_h)
        print()
        get_watchlist_prices(watchlist)
        print()
        print()
        print("Note: You may enter a menu number at any time.")
        print()

        # Ask user to enter a menu option.
        choice = input("Enter a menu option or Watchlist ticker: ")
        menu_options_2(choice, False, username)

        # Selecting 'Guide' results in resetting the screen without error message.
        screen_clear()
        home_screen(username)


def get_watchlist(username):
    """A function that opens and reads watchlist.txt, and populates a list with the
    user's current watchlist stocks."""

    watchlist = []
    with open('watchlist.txt', 'r') as file:
        for line in file:
            if line.startswith(username):
                ticker = line.split(':')[1].strip()
                watchlist.append(ticker)
    return sorted(watchlist)


def get_watchlist_prices(watchlist):
    """A function that retrieves current stock prices via the get_stock_data function,
    and then displays the user's watchlist stocks and their associated current prices."""

    # Iterate through the user's Watchlist.
    for ticker in watchlist:
        data = get_stock_data(ticker)

        # Display current price for each ticker (if statements for formatting consistency).
        if len(ticker) == 1:
            print(f"{ticker}...............................${data['current_price']:.2f}")
        elif len(ticker) == 2:
            print(f"{ticker}..............................${data['current_price']:.2f}")
        elif len(ticker) == 3:
            print(f"{ticker}.............................${data['current_price']:.2f}")
        elif len(ticker) == 4:
            print(f"{ticker}............................${data['current_price']:.2f}")
        else:
            print(f"{ticker}...........................${data['current_price']:.2f}")


# ----------------------------------------------------------------------------------------------------------------------
# STOCK SEARCH SCREEN FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def stock_search_screen(username):
    """A function that displays the 'Stock Search' screen, and awaits user input to
    initiate menu navigation or navigation to the chosen ticker's stock screen."""

    # Display the 'Stock Search' screen.
    while True:
        screen_clear()
        print(Fore.MAGENTA + Style.BRIGHT + stock_prophet + Style.RESET_ALL)
        print(stocks_m)
        print()
        print(search_h)
        print()
        print("- For list of stock tickers...")
        print()
        print("     1) Open the Guide")
        print("     2) Go to 'Additional Information'")
        print("     3) Click 'Stock Tickers'")
        print()
        print()
        print("Note: You may enter a menu number at any time.")
        print()

        # Ask user to enter a stock ticker or menu option.
        choice = input("Enter a stock ticker (or menu option): ")

        # Check if user entered a menu option.
        menu_options_2(choice, True, username)

        # Retrieve list of tickers, then check if user entered a valid stock ticker.
        ticker_list = get_ticker_list()
        ticker_check(choice.upper(), ticker_list, username)


def get_ticker_list():
    """A function that returns a list of valid stock tickers."""

    # Open and read tickers.txt.
    with open("tickers.txt", "r") as file:
        ticker_symbols = file.readlines()

    # Populate and return list of tickers.
    tickers = [ticker.strip() for ticker in ticker_symbols]
    return tickers


def ticker_check(ticker, ticker_list, username):
    """A function that checks whether the user entered a valid ticker, and if so,
    redirects the user to the ticker's specific stock screen."""

    # Print statements for enhancing UI design.
    print(".")
    print(".")
    print(".")
    print(".")

    # Check if the user-entered ticker is in the list.
    if ticker in ticker_list:
        print(f"Ticker '{ticker}' found!")
        sleep(3)
        screen_clear()
        stock_screen(username, ticker)
    elif ticker == "3":  # Selecting 'Guide' results in resetting the screen without error message.
        screen_clear()
        stock_search_screen(username)
    else:
        print(f"Ticker '{ticker}' not found. Please try again")
        sleep(3)
        screen_clear()
        stock_search_screen(username)


# ----------------------------------------------------------------------------------------------------------------------
# STOCK SCREEN FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def stock_screen(username, ticker):
    """A function that displays the 'Stock' screen, initializes the ticker's stock
    info, and awaits user input to initiate menu navigation or process a command."""

    # Get and initialize stock data.
    data = get_stock_data(ticker)

    # Display the 'Stock' screen.
    while True:
        screen_clear()
        print(Fore.MAGENTA + Style.BRIGHT + stock_prophet + Style.RESET_ALL)
        print(stocks_m)
        print()
        print()
        print(Fore.MAGENTA + stock_h.renderText("$" + ticker.upper() + "      " + "$" + f"{data['current_price']:.2f}"))
        print(Style.RESET_ALL + f"COMPANY......................................{data['company_name']}")
        print(f"INDUSTRY.....................................{data['industry']}")
        print(f"EXCHANGE.....................................{data['exchange']}")
        print(f"MARKET CAP...................................{'${:,.0f}'.format(data['market_cap'])}")
        print(f"DAILY VOLUME.................................{'{:,.0f}'.format(data['daily_volume'])}")
        print(f"AVERAGE VOLUME...............................{'{:,.0f}'.format(data['average_volume'])}")
        print()
        print()
        print(Fore.MAGENTA + commands_h + Style.RESET_ALL)
        print("(Open Guide for walkthrough and more info!)")
        print()
        print()
        print("Enter 'A' --> Add stock to your Watchlist")
        print("Enter 'B' --> Remove stock from your Watchlist")
        print("Enter 'C' --> View 1 month stock chart")
        print("Enter 'D' --> Get 1 month price prediction")
        print()
        print()
        print()
        print("Note: You may enter a menu number at any time.")
        print()

        # Ask user to enter a command.
        command = input("Enter a command (or menu option): ")

        # Check if user entered a valid command or a menu option.
        menu_options_2(command, True, username)
        command_check(command, username, ticker)


def command_check(command, username, ticker):
    """A function that checks whether the user entered a valid command, and if so,
    calls the chosen command's particular function(s)."""

    # Check if user-entered command is valid.
    if command.upper() == "A":
        confirm_command_a(username, ticker)
    elif command.upper() == "B":
        confirm_command_b(username, ticker)
    elif command.upper() == "C":
        one_month_chart(ticker)
    elif command.upper() == "D":
        process_prediction(ticker, 30)
    elif command == "3":  # Selecting 'Guide' results in resetting the screen without error message.
        screen_clear()
        stock_screen(username, ticker)
    else:
        print()
        print("Error: You did not enter a valid command/option. Please try again.")
        sleep(3)

    # Reset the screen after processing command or returning Error.
    screen_clear()
    stock_screen(username, ticker)


def confirm_command_a(username, ticker):
    """A function that prompts users to confirm their choice of command "A", and if
    confirmed, calls the add_to_watchlist function."""

    # Ask user to confirm.
    print()
    confirm = input(f"Are you sure you want to add '{ticker}' to your watchlist? (Y/N): ")
    if confirm.upper() == 'Y':
        add_to_watchlist(username, ticker)


def add_to_watchlist(username, ticker):
    """A function that adds the given ticker to the user's watchlist."""

    # Check if ticker is already in the watchlist, otherwise, add to watchlist.
    with open('watchlist.txt', 'r+') as file:
        watchlist = file.read()
        if f"{username}:{ticker}" in watchlist:
            print()
            print(f"Ticker '{ticker}' is already in your watchlist.")
        else:
            file.write(f"{username}:{ticker}\n")
            print()
            print(f"Ticker '{ticker}' added to {username}'s watchlist.")
    sleep(2)


def confirm_command_b(username, ticker):
    """A function that prompts users to confirm their choice of command "B", and if
    confirmed, calls the remove_from_watchlist function."""

    # Ask user to confirm.
    print()
    confirm = input(f"Are you sure you want to remove '{ticker}' from your watchlist? (Y/N): ")
    if confirm.upper() == 'Y':
        attempt_status = remove_from_watchlist(username, ticker)

        # Inform user whether the ticker was removed.
        if attempt_status:
            print()
            print(f"Ticker '{ticker}' removed from {username}'s watchlist.")
        else:
            print()
            print(f"Ticker '{ticker}' is not in your watchlist.")
        sleep(2)


def remove_from_watchlist(username, ticker):
    """A function that removes the given ticker from the user's watchlist."""

    # Rewrite watchlist.txt, skipping over the given ticker if present.
    with open('watchlist.txt', 'r') as file:
        lines = file.readlines()
    removed = False
    with open('watchlist.txt', 'w') as file:
        for line in lines:
            if not (line.startswith(username) and line.strip().endswith(ticker)):
                file.write(line)
            else:
                removed = True

    # Return attempt status.
    return removed


def one_month_chart(ticker):
    """A function that plots a stock ticker's closing prices from the last 30
    days, and then displays the chart."""

    # Initialize stock data.
    stock_data = yf.download(ticker, start="2024-02-08", end="2024-03-08")

    # Plot the closing prices.
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label='Close Price', color='blue')
    plt.title(f"{ticker} Stock Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Display the chart.
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# STOCK PROPHET HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def screen_clear():
    """A function that clears the user's terminal screen."""

    # Identify user's operating system for screen clearing.
    if os.name == 'posix':  # Linux/Mac
        clear = 'clear'
    else:
        clear = 'cls'  # Windows

    # Clear user's screen.
    os.system(clear)


def menu_options_1(user_input, check):
    """A function that processes menu choices for the Start, Sign Up,
    and Login screens."""
    if user_input == "1":
        screen_clear()
        start_screen()
    elif user_input == "2":
        screen_clear()
        signup_screen()
    elif user_input == "3":
        screen_clear()
        login_screen()
    elif user_input == "4":
        screen_clear()
        print(exit_h)
        sleep(2)
        screen_clear()
        exit()
    elif check is False:
        print("Error: You did not enter a valid menu option. Please try again.")
        sleep(3)
        screen_clear()
        start_screen()


def menu_options_2(user_input, check, username):
    """A function that processes menu choices for the Home, Stock Search,
    and Stock screens."""
    if user_input == "1":
        screen_clear()
        home_screen(username)
    elif user_input == "2":
        screen_clear()
        stock_search_screen(username)
    elif user_input == "3":
        file = 'Guide.pdf'
        file_path = os.path.abspath(file)
        if os.name == 'posix':
            open_file_unix(file_path)
        else:
            open_file_windows(file_path)
    elif user_input == "4":
        screen_clear()
        print(exit_h)
        sleep(2)
        screen_clear()
        start_screen()
    elif check is False:
        if user_input.upper() in get_watchlist(username):
            screen_clear()
            stock_screen(username, user_input)
        else:
            print()
            print("Error: You did not enter a valid number or ticker. Please try again.")
            sleep(3)
            screen_clear()
            home_screen(username)


def open_file_windows(file_path):
    """A function that receives a Windows-based file path and opens the file."""
    os.startfile(file_path)


def open_file_unix(file_path):
    """A function that receives a Unix-based file path and opens the file."""
    subprocess.run(['open', file_path])


def get_stock_data(ticker):
    """A function that sends HTTP GET requests to receive and return a given
    ticker's stock data."""

    # URL to my CS-361 partner's microservice.
    url = f"https://daugherc.pythonanywhere.com/stock?ticker={ticker}"

    # Send GET request and initialize to variable.
    get_request = requests.get(url=url, params=ticker)

    # Extract data in JSON and return.
    stock_data = get_request.json()
    return stock_data


# ----------------------------------------------------------------------------------------------------------------------
# RUN STOCK PROPHET
# ----------------------------------------------------------------------------------------------------------------------

# Start colorama.
init()

# Filter warnings.
warnings.filterwarnings("ignore")

# Clear terminal and display 'Start' screen.
screen_clear()
start_screen()
