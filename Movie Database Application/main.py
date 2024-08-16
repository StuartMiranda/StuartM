import mysql.connector
from mysql.connector import OperationalError
from mysql.connector import Error
import os
from getpass4 import getpass
import shutil
import time
from tabulate import tabulate
from datetime import datetime

# -----------------------------------
# Utility Functions
# -----------------------------------
cwd = os.getcwd() # get working directory
scripts = cwd + "\\scripts\\" # get scripts directory
max_length = 30

def cut_and_wrap(text):
        if len(text) > max_length:
            return '\n'.join(text[:max_length].split(' ')[:-1]) + "..."
        else:
            return text

# run external sql scripts
def executeSQL(filename):
    # Open and read the file
    fd = open(filename, 'r')
    sqlScript = fd.read()
    fd.close()

    # all SQL commands (split on ';')
    sqlCommands = sqlScript.split(';')

    # Execute every command from the input file
    for command in sqlCommands:
        # skip and print commands that fail
        try:
            c.execute(command)
        except OperationalError as msg:
            print("Command skipped: ", msg)

# clears terminals for CLI purposes
def clear_terminal():
    # Check if the system is Windows or Unix/Linux/Mac
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# copy prod/sample tables to correct directory for db loading
def copy_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)

        if os.path.isfile(src_file):
            try:
                shutil.copy(src_file, dest_file)
                print(f"Successfully copied {src_file} to {dest_file}")
            except Exception as e:
                print(f"Failed to copy {src_file} to {dest_file}: {e}")

def paginate(data, page_size):
    total_rows = len(data)
    for start in range(0, total_rows, page_size):
        end = min(start + page_size, total_rows)
        page = data[start:end]
        print(tabulate(page, headers="firstrow", tablefmt="grid"))
        if end < total_rows:
            #input("Press Enter to continue...")
            print("next page")

# rebuild database
def redo_database():
    print("Do you want to rewrite the whole database?")
    print("(you must do this if you haven't created the DB yet)")
    redo = input("y/[n]: ")

    if (redo == "y"):
        print("1: Use sample dataset")
        print("2: Use full prod dataset")
        dbtype = input("([1]/2): ")

        print("remaking database")
        # create db if does not exist
        c.execute("DROP DATABASE IF EXISTS netflix")
        c.execute("CREATE DATABASE netflix")

        # create tables
        # print("making tables")
        executeSQL(scripts + 'Create Tables v4.sql')


        # COPY DATA TO UPLOADS
        print("copying csv files")
        sqlpath = 'C:\\ProgramData\\MySQL\\MySQL Server 8.4\\Uploads'
        table_src = cwd + "\\tables"

        if dbtype == "2":
            table_src = table_src + "\\prod"
        else:
            table_src = table_src + "\\sample"

        copy_files(table_src, sqlpath)

        print("Loading data into database... (this may take a while)")
        start_time = time.perf_counter()

        # load data
        executeSQL(scripts + 'load_data v3.sql')
        netflixdb.commit()

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Done! Time taken: {execution_time:.6f} seconds")

# tabulates result, a resulting sql query
def print_table(table_data):
    table_headers = [col[0] for col in c.description]
    table_data = list(map(cut_and_wrap, table_data))

    formatted_table = tabulate(table_data, headers=table_headers, tablefmt="grid")
    print(formatted_table)



# -----------------------------------
# R6 Functions
# -----------------------------------

# sample search function. Returns movie ID in a table
def search_title():
    search = input("Input search Query:") #user input

    # query
    query = "SELECT title, movie_id from movie where title like %s ORDER BY title LIMIT 30"

    # run and print query
    c.execute(query, ('%' + search + '%',))
    result = c.fetchall()
    print_table(result)

# takes string and prints new lines if string is too long
def print_summary(summary):
    print("Movie Summary:")
    words = summary.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) <= 50:
            current_line += word + " "
        else:
            print(current_line.rstrip())
            current_line = word + " "
    print(current_line.rstrip())
    print("")


# genre dictionary for searching genres
genre_dict = {
    1: "Adventure",
    2: "Fantasy",
    3: "Animation",
    4: "Drama",
    5: "Horror",
    6: "Action",
    7: "Comedy",
    8: "History",
    9: "Western",
    10: "Thriller",
    11: "Crime",
    12: "Documentary",
    13: "Science Fiction",
    14: "Mystery",
    15: "Music",
    16: "Romance",
    17: "Family",
    18: "War",
    19: "Foreign",
    20: "TV Movie"
}

genre_list = list(genre_dict.items()) # Convert the dictionary to a list of tuples

# Split the list into two halves
half = len(genre_list) // 2
first_half = genre_list[:half]
second_half = genre_list[half:]

# Make the halves the same length by adding empty entries if needed
if len(first_half) > len(second_half):
    second_half.append((None, None))

# Combine the halves side by side
combined_list = []
for (id1, name1), (id2, name2) in zip(first_half, second_half):
    combined_list.append([id1, name1, id2, name2])

genre_list_print = tabulate(combined_list, headers=["Genre ID", "Genre Name", "Genre ID", "Genre Name"], tablefmt="grid") # Print the table of genres

# given a string selection (potentially separated by commas), search up and return genre names from genre_dict as a list
def select_genre(genre_numbers_str, genre_dict):
        # Check if the input string is empty
    if not genre_numbers_str.strip():
        return []

    # Split the input string by commas and strip any extra spaces
    genre_numbers_str_list = genre_numbers_str.split(',')
    
    genre_numbers = []

    # check if all separated strings are integers
    for num_str in genre_numbers_str_list:
        num_str = num_str.strip()
        if num_str:
            try:
                genre_numbers.append(int(num_str))
            except ValueError:
                print(f"Error: '{num_str}' is not a valid integer. Exiting...")
                return []

    # Look up the genre names using the genre_dict
    genre_names = [genre_dict[num] for num in genre_numbers if num in genre_dict]

    return genre_names

# takes a list of genre names and formats them properly for use in a sql query
def format_genre_query(genre_names):
    # Join the genre names into a single string formatted for SQL
    return ', '.join(f"'{genre}'" for genre in genre_names)

# main genre search function
def search_genre():

    # select genre
    print(genre_list_print)
    print("Select the following genres you would like to search (search multiple, separated by commas. Ignores non valid selections):")
    print("Example: 6,9 --> action, western")
    genres = input("Selection:") #user input

    # get formatted genre names for query
    genre_names = select_genre(genres, genre_dict)
    formated_genres = format_genre_query(genre_names)

    # exit if nothing is entered, or is not properly formatted
    if not formated_genres:
        print("Invalid entry detected (most likely genre ID not found)")
        return

    print("Movies with genres:" + formated_genres)
    #query
    query = '''
        SELECT DISTINCT m.title , m.vote_average
        FROM movie m 
        JOIN Classified_In c ON m.movie_id = c.movie_id
        JOIN genre g ON c.genre_id = g.genre_id 
        WHERE g.genre_name IN (''' + formated_genres + ''')
        ORDER BY vote_average
        DESC LIMIT 20;'''
    
    # run query
    c.execute(query)
    result = c.fetchall()

    # if no movies found, exit
    if not result:
        print("No movie(s) found...")
        return

    # else print table
    print_table(result)

def search_credited_person():
    person = input("Search person name fuzzy (exits if no input):")

    if person.strip() == "":
        print("no input detected, exiting...")
        return

    query = '''
        SELECT m.title as 'Featured Title'
        FROM movie m
        JOIN Comprises_Of c ON m.movie_id = c.movie_id
        JOIN credit cr ON c.person_id = cr.person_id
        WHERE cr.name LIKE %s
        LIMIT 30;
        '''
    c.execute(query, ('%' + person + '%',))
    result = c.fetchall()

    if not result:
        print("No movie(s) found...")
        return
    
    print_table(result)

# sample search function. Returns movie ID in a table
def movie_data_simple():
    movie_id = input("Input movie ID:") # get user input

    # get movie ID info
    query = '''
        SELECT title as Title, release_date as "Release date", status as Status, runtime as "Runtime (Minutes)", adult as "R Rated?", vote_average as "Avg. Rating", vote_count as "Rating count"
        from movie 
        where movie_id = %s'''
    c.execute(query, (movie_id,))
    result = c.fetchall()

    # if nothing is selected (i.e., movie id is not found), exit
    if not result:
        print("No movie found...")
        return

    # print result
    print_table(result)

    # print movie summary
    query = "SELECT overview from movie where movie_id = %s"
    c.execute(query, (movie_id,))
    summary = c.fetchall()
    summary = summary[0]
    summary = summary[0]

    print_summary(summary)

# print genre, production and cast info
def move_data_full():
    movie_id = input("Input movie ID:")

    #queries
    test_query = '''
        SELECT *
        FROM movie m
        WHERE m.movie_id = %s
    '''
    genre_query = '''
        SELECT g.genre_name AS Genre
        FROM Genre g 
        JOIN Classified_In c ON g.genre_id = c.genre_id 
        JOIN movie m ON c.movie_id = m.movie_id 
        WHERE m.movie_id = %s
        '''
    production_query = '''
        SELECT p.production_name AS Productions 
        FROM Production p 
        JOIN Produced_By p1 ON p1.production_id = p.production_id 
        JOIN movie m ON p1.movie_id = m.movie_id 
        WHERE m.movie_id = %s
        '''
    credits_query = '''
        SELECT cr.name AS Name, GROUP_CONCAT(cc.character SEPARATOR ', ') as Characters, NULL AS Roles
        FROM movie m
        JOIN comprises_of co ON m.movie_id = co.movie_id
        JOIN credit cr ON co.person_id = cr.person_id
        JOIN cast cc ON cr.person_id = cc.person_id AND m.movie_id = cc.movie_id
        WHERE m.movie_id = %s
        GROUP BY cr.name

        UNION

        SELECT cr.name AS Name, NULL AS Characters, GROUP_CONCAT(p.Job_Name SEPARATOR ', ') as Roles
        FROM movie m
        JOIN comprises_of co ON m.movie_id = co.movie_id
        JOIN credit cr ON co.person_id = cr.person_id
        JOIN performs p ON cr.person_id = p.person_id AND m.movie_id = p.movie_id
        WHERE m.movie_id = %s
        GROUP BY cr.name

        '''
    
    c.execute(test_query, (movie_id,))
    test_result = c.fetchall()

    if not test_result:
        print("Movie not found...")
        return

    c.execute(genre_query, (movie_id,))
    genre_result = c.fetchall()

    print_table(genre_result)

    c.execute(production_query, (movie_id,))
    prod_result = c.fetchall()

    print_table(prod_result)

    c.execute(credits_query, (movie_id, movie_id))
    creds_result = c.fetchall()

    table_headers = [col[0] for col in c.description]
    table_data = creds_result

    #print(credits_table)
    paginate(table_data, 20)

    return


# -----------------------------------
# R7 Functions
# -----------------------------------

def search_production():
    production = input("Input production Query (no input exits):")

    if production.strip() == "":
        print("no input detected, exiting...")
        return

    query = '''
        SELECT m.movie_id, m.title, p.production_name
        FROM movie m
        JOIN Produced_By c ON m.movie_id = c.movie_id
        JOIN Production p ON c.production_id = p.production_id
        WHERE production_name LIKE %s
        ORDER BY p.production_name, m.movie_id, m.release_date DESC
        LIMIT 30
        '''
    c.execute(query, ('%' + production + '%',))
    result = c.fetchall()

    if not result:
        print("No Movie(s) found...")
        return

    print_table(result)

# -----------------------------------
# R8 Functions
# -----------------------------------

def update_movie():
    # get movie id to edit
    movie_id = input("Input movie ID:")
    query = '''
        SELECT movie_id, title, status, release_date, adult, video, runtime
        from movie 
        where movie_id = %s
        '''
    c.execute(query, (movie_id,))
    result = c.fetchall()

    # quit if no movie found
    if not result:
        print("Movie not found, exiting...")
        return

    print_table(result)

    # print summary
    query = "SELECT overview from movie where movie_id = %s"
    c.execute(query, (movie_id,))
    summary = c.fetchall()
    summary = summary[0]
    summary = summary[0]

    print_summary(summary)

    # print update options
    print("1: Movie ID")
    print("2: Title")
    print("3: Overview")
    print("4: Status")
    print("5: Release Date")
    print("6: Adult")
    print("7: video")
    print("8: Runtime")
    print("9: Exit to previous menu")
    user = input("Select movie_attribute to edit (Default 9): ")

    #movie id
    if user == "1":
        print("Type nothing and press enter to cancel")
        new_movie_id = input("Input new movie ID (non-negative int, exits on no input): ")
        if not new_movie_id.strip():
            clear_terminal()
            print("No input detected, exiting...")
            return

        # check if new movie id is non-negative int
        if not new_movie_id.isdigit() or new_movie_id.isdigit() < 0 or new_movie_id.strip() == "":
            clear_terminal()
            print("Invalid input, must be non-negative integer, exiting...")
            return
        
        # check if new movie_id already exists as a movie
        check_unique_id = '''
            SELECT *
            FROM movie
            where movie_id = %s
            '''
        c.execute(query, (new_movie_id,))
        check_result = c.fetchall()

        if check_result:
            clear_terminal()
            print("Movie ID already exists, exiting...")
            return

        query = '''
            UPDATE movie
            SET movie_id = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_movie_id,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")

    #title
    elif user == "2":
        print("Type nothing and press enter to cancel")
        new_title = input("Input new movie title: ")
        if not new_title.strip():
            clear_terminal()
            print("No input detected, exiting...")
            return
        
        query = '''
            UPDATE movie
            set title = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_title,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")
        
    #Overview
    elif user == "3":
        print("Type nothing and press enter to cancel")
        new_overview = input("Input new overview: ")
        
        query = '''
            UPDATE movie
            set overview = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_overview,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")

    #Status
    elif user == "4":
        options = {
            1: "Rumored",
            2: "Planned",
            3: "In Production",
            4: "Post Production",
            5: "Released",
            6: "Canceled",
            7: "Exit",
        }

        print("1: Rumored")
        print("2: Planned")
        print("3: In Production")
        print("4: Post Production")
        print("5: Released")
        print("6: Canceled")
        print("7: Exit")
        
        new_status = input("Select new status (default 7): ")
        try:
            choice = int(new_status)
            if choice in options:
                new_status = options[choice]
            else:
                clear_terminal()
                print("Invalid input, exiting...")
                return
        except ValueError:
            clear_terminal()
            print("Invalid input, exiting...")
            return

        print(new_status)
        query = '''
            UPDATE movie
            set status = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_status,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")

    # release date
    elif user == "5":
        print("Type nothing and press enter to cancel")
        new_date = input("Input new overview (yyyy-mm-dd format): ")
        if not new_date.strip():
            clear_terminal()
            print("Invalid date format, exiting...")
            return
        
        try:
            datetime.strptime(new_date, "%Y-%m-%d")
            print("Valid Date")
        except ValueError:
            clear_terminal()
            print("Invalid date format, exiting...")
            return
        
        query = '''
            UPDATE movie
            set release_date = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_date,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")

    #Adult
    elif user == "6":
        options = {
            1: "True",
            2: "False",
            3: "Exit",
        }

        print("1: True")
        print("2: False")
        print("3: Exit")
        
        new_adult = input("Select option (default 3): ")
        try:
            choice = int(new_adult)
            if choice in options:
                new_adult = options[choice]
            else:
                clear_terminal()
                print("Invalid input, exiting...")
                return
        except ValueError:
            clear_terminal()
            print("Invalid input, exiting...")
            return
        
        query = '''
            UPDATE movie
            set adult = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_adult,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")

    #video
    elif user == "7":
        options = {
            1: "True",
            2: "False",
            3: "Exit",
        }

        print("1: True")
        print("2: False")
        print("3: Exit")
        
        new_video = input("Select option (default 3): ")
        try:
            choice = int(new_video)
            if choice in options:
                new_video = options[choice]
            else:
                clear_terminal()
                print("Invalid input, exiting...")
                return
        except ValueError:
            clear_terminal()
            print("Invalid input, exiting...")
            return
        
        query = '''
            UPDATE movie
            set video = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_video,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")

    #runtime
    elif user == "8":
        print("Type nothing and press enter to cancel")
        new_runtime = input("Input new runtime, in minutes (non-negative int): ")
        if not new_runtime.strip():
            clear_terminal()
            print("No input detected, exiting...")
            return

        # check if new movie id is non-negative int
        if not new_runtime.isdigit() or new_runtime.isdigit() < 0:
            clear_terminal()
            print("Invalid input, must be non-negative integer, exiting...")
            return

        query = '''
            UPDATE movie
            SET runtime = %s
            WHERE movie_id = %s
            '''
        c.execute(query, (new_runtime,movie_id))
        netflixdb.commit()
        clear_terminal()
        print("successful!")
    else:
        return


# insert movie
def insert_movie():
    # get new movie attributes

    # movie id
    movie_id = input("Enter movie_id: ")

    # check if nothing was inputed
    if not movie_id.strip():
            clear_terminal()
            print("No input detected, exiting...")
            return

    # check if new movie id is non-negative int
    if not movie_id.isdigit() or movie_id.isdigit() < 0 or movie_id.strip() == "":
        clear_terminal()
        print("Invalid input, must be non-negative integer, exiting...")
        return
    
    # check if new movie_id already exists as a movie
    check_unique_id = '''
        SELECT *
        FROM movie
        where movie_id = %s
        '''
    c.execute(check_unique_id, (movie_id,))
    check_result = c.fetchall()
    
    if check_result:
        clear_terminal()
        print("Movie ID already exists, exiting...")
        return



    # get title
    title = input("Enter Title: ")

    if not title.strip():
            clear_terminal()
            print("No input detected, exiting...")
            return


    # get overview
    overview = input("Enter overview: ")

    if not overview.strip():
            clear_terminal()
            print("No input detected, exiting...")
            return

    # get status
    options = {
            1: "Rumored",
            2: "Planned",
            3: "In Production",
            4: "Post Production",
            5: "Released",
            6: "Canceled",
            7: "Exit",
        }

    print("1: Rumored")
    print("2: Planned")
    print("3: In Production")
    print("4: Post Production")
    print("5: Released")
    print("6: Canceled")
    print("7: Exit")
        
    status = input("Select new status (default 7): ")
    try:
        choice = int(status)
        if choice in options:
            status = options[choice]
        else:
            clear_terminal()
            print("Invalid input, exiting...")
            return
    except ValueError:
        clear_terminal()
        print("Invalid input, exiting...")
        return
    
    print("Type nothing and press enter to cancel")
    release_date = input("Input new overview (yyyy-mm-dd format): ")
    if not release_date.strip():
        clear_terminal()
        print("Invalid date format, exiting...")
        return
    
    try:
        datetime.strptime(release_date, "%Y-%m-%d")
        print("Valid Date")
    except ValueError:
        clear_terminal()
        print("Invalid date format, exiting...")
        return
    

    # adult
    options = {
            1: "True",
            2: "False",
            3: "Exit",
        }

    print("1: True")
    print("2: False")
    print("3: Exit")
    
    adult = input("Select new adult value (default 3): ")
    try:
        choice = int(adult)
        if choice in options:
            adult = options[choice]
        else:
            clear_terminal()
            print("Invalid input, exiting...")
            return
    except ValueError:
        clear_terminal()
        print("Invalid input, exiting...")
        return

    # get video
    print("1: True")
    print("2: False")
    print("3: Exit")
    
    video = input("Select new video value (default 3): ")
    try:
        choice = int(video)
        if choice in options:
            video = options[choice]
        else:
            clear_terminal()
            print("Invalid input, exiting...")
            return
    except ValueError:
        clear_terminal()
        print("Invalid input, exiting...")
        return
    
    # new runtime
    runtime = input("Input new runtime, in minutes (non-negative int): ")
    if not runtime.strip():
        clear_terminal()
        print("No input detected, exiting...")
        return

    # check if new movie id is non-negative int
    if not runtime.isdigit() or runtime.isdigit() < 0:
        clear_terminal()
        print("Invalid input, must be non-negative integer, exiting...")
        return
    


    # query
    query = '''
        INSERT INTO Movie (movie_id, title, overview, status, release_date, adult, video, runtime, vote_average, vote_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, "0", "0")
        '''
    
    try:
        c.execute(query, (movie_id, title, overview, status, release_date, adult, video, runtime))
        netflixdb.commit()
        print("Movie inserted!")
    except Error as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

def delete_movie():
    movie_id = input("Input movie ID:")
    query = '''
        SELECT *
        from movie 
        where movie_id = %s
        '''
    c.execute(query, (movie_id,))
    result = c.fetchall()

    # quit if no movie found
    if not result:
        print("Movie not found, exiting...")
        return
    
    delete_query = '''
        DELETE
        FROM Movie
        WHERE movie_id = %s
        '''
    
    try:
        c.execute(delete_query, (movie_id,))
        result = c.fetchall()
        netflixdb.commit()
        print("Succesfully deleted movie!")
    except Error as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# -----------------------------------
# R9 Functions
# -----------------------------------
def success_rate_per_crew():

    # setting to let query run
    set_query = "SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));"

    # actual query
    query = """
        SELECT cr.name as Name, j.Job_Name as Job, AVG(m.vote_average) AS "Avg. Rating"
        FROM Movie m 
        JOIN Classified_In ci ON m.movie_id = ci.movie_id 
        JOIN Genre g ON ci.genre_id = g.genre_id 
        JOIN Performs p ON p.movie_id = m.movie_id 
        JOIN Crew c ON p.person_id = c.person_id 
        JOIN Job j ON p.Job_Name = j.Job_Name 
        JOIN Credit cr ON c.person_id = cr.person_id
        GROUP BY cr.name
        ORDER BY "Avg. Rating" DESC
        LIMIT 20;
       """
    c.execute(set_query)
    c.execute(query)
    result = c.fetchall()

    print_table(result)


# -----------------------------------
# R10 Functions
# -----------------------------------

# map genres to IDs for conversion and query
genre_mapping = {
    1: 12,   # Adventure
    2: 14,   # Fantasy
    3: 16,   # Animation
    4: 18,   # Drama
    5: 27,   # Horror
    6: 28,   # Action
    7: 35,   # Comedy
    8: 36,   # History
    9: 37,   # Western
    10: 53,  # Thriller
    11: 80,  # Crime
    12: 99,  # Documentary
    13: 878, # Science Fiction
    14: 9648,# Mystery
    15: 10402,# Music
    16: 10749,# Romance
    17: 10751,# Family
    18: 10752,# War
    19: 10769,# Foreign
    20: 10770 # TV Movie
}

# R10 main function
def best_production_studios():
    # print genres and get ID from user
    print(genre_list_print)
    genre_id = input("Select a genre: ")

    # convert choice to int and check if valid, else exit
    try:
        choice = int(genre_id)
        if choice in genre_mapping:
            #print(f"The genre ID for your selection is: {genre_mapping[choice]}")
            choice = genre_mapping[choice]
        else:
            print("Invalid input, exiting...")
            return
    except ValueError:
        print("Invalid input, exiting...")
        return

    # query
    query = """
        SELECT p.production_name, AVG(m.vote_average) AS vote_score  
        FROM Movie m 
        JOIN Classified_In ci ON m.movie_id = ci.movie_id 
        JOIN Genre g ON ci.genre_id = g.genre_id 
        JOIN Produced_By pb ON m.movie_id = pb.movie_id 
        JOIN Production p ON pb.production_id = p.production_id 
        WHERE g.genre_id = %s
        GROUP BY p.production_name ORDER BY vote_score DESC
        LIMIT 20;
        """
    
    #execute query and print table
    c.execute(query, (choice,))
    result = c.fetchall()

    print_table(result)

# -----------------------------------
# R11 Functions
# -----------------------------------

# counts the number of movies that each genre has
def genre_count():
    # query
    query = '''
        SELECT g.genre_name as Genre, COUNT(*) AS "Movie Count" 
        FROM genre g
        JOIN Classified_In c ON g.genre_id = c.genre_id 
        GROUP BY g.genre_name 
        ORDER BY "Movie Count" DESC
        LIMIT 10
        '''
    
    # run and print query result
    c.execute(query)
    result = c.fetchall()
    print_table(result)

# averages ratings for each respective genre
def best_genre():
    # query
    query = '''
        SELECT g.genre_name as Genre, ROUND(AVG(m.vote_average), 2) AS "Avg. Rating"
        FROM genre g JOIN Classified_In c ON g.genre_id = c.genre_id
        JOIN movie m ON c.movie_id = m.movie_id
        GROUP BY g.genre_name
        ORDER BY "Avg. Rating" DESC;
        '''
    
    # run and print query result
    c.execute(query)
    result = c.fetchall()
    print_table(result)

# -----------------------------------
# R13  Functions
# -----------------------------------


def advanced_search():

    #init variables
    release_date_query = ""
    runtime_query = ""
    vote_average_query = ""
    vote_count_query = ""
    subquery = []

    while True:
        print("Current Filters [current search query]")

        # print options
        print("1: release date [" + release_date_query +"]")
        print("2: runtime [" + runtime_query +"]")
        print("3: vote_average [" + vote_average_query +"]")
        print("4: vote count [" + vote_count_query +"]")
        print("5: submit search")
        print("6: exit")

        user = input("Select filter options:") 

        # handle user filter selections

        # release date
        if user == "1":
            operator = ""
            release_date_x = ""
            release_date_y = ""

            print("1: release date after x")
            print("2: release date before x ")
            print("3: release date between x and y (x <= y)")
            print("4: turn filter off")
            print("5: leave filter as is")

            release_user = input("input option (default 5):")

            # pick operator
            if release_user == "1":
                operator = ">="
            if release_user == "2":
                operator = "<="
            
            if release_user == "4":
                release_date_query = ""
            elif release_user == "5":
                #placeholder, does nothing and exits
                operator = ""
            elif release_user == "3":
                release_date_x = input("Input date x (yyyy-mm-dd format): ")
                if not release_date_x.strip():
                    clear_terminal()
                    print("Invalid date format, exiting...")
                    return
                
                try:
                    datetime.strptime(release_date_x, "%Y-%m-%d")
                    print("Valid Date")
                except ValueError:
                    clear_terminal()
                    print("Invalid date format, exiting...")

                # make query for options 1 and 2
                if release_user == "1" or release_user == "2":
                    release_date_query = "release_date " + operator + " '" + release_date_x + "'"
                    # release_date <=/>= 'date_x'
                
                
                # release date between x and y
                # get second date
                if release_user == "3":
                    release_date_y = input("Input date y (yyyy-mm-dd format): ")
                    if not release_date_y.strip():
                        clear_terminal()
                        print("Invalid date format, exiting...")
                        return
                    
                    try:
                        datetime.strptime(release_date_y, "%Y-%m-%d")
                        print("Valid Date")
                    except ValueError:
                        clear_terminal()
                        print("Invalid date format, exiting...")

                    # check if x <= y
                    if not (release_date_x <= release_date_y):
                        print("Error: x is not less than or equal to y")
                        return

                    release_date_query = "release_date BETWEEN '" + release_date_x + "' AND '" + release_date_y + "'"
                    # release_date BETWEEN 'date_x' AND 'date_y'
            else:
                operator = ""
                # does nothing

        #runtime
        if user == "2":
            runtime_x = ""
            runtime_y = ""
            operator = ""
            print("1: runtime > x")
            print("2: runtime < x ")
            print("3: runtume between x and y (x <= y)")
            print("4: turn filter off")
            print("5: leave filter as is")

            runtime_user = input("input option (default 5):")

            # pick operator
            if runtime_user == "1":
                operator = ">="
            if runtime_user == "2":
                operator = "<="
            
            if runtime_user == "4":
                runtime_query = ""
            elif runtime_user == "5":
                #placeholder, does nothing and exits
                operator = ""
            elif runtime_user =="3":
                runtime_x = input("Input runtime x (non-negative int): ")

                # check if int and non-negative
                if not runtime_x.strip():
                    clear_terminal()
                    print("No input detected, exiting...")
                    return

                # check if new movie id is non-negative int
                if not runtime_x.isdigit() or runtime_x.isdigit() < 0:
                    clear_terminal()
                    print("Invalid input, must be non-negative integer, exiting...")
                    return

                # make query for options 1 and 2
                if runtime_user == "1" or runtime_user == "2":
                    runtime_query = "runtime " + operator + " '" + runtime_x + "'"
                    # runtime <=/>= 'runtime_x'
                
                
                # release date between x and y
                # get second date
                if runtime_user == "3":
                    runtime_y = input("Input runtime y (on-negative int): ")
                    
                    # check if int and non-negative
                    if not runtime_y.strip():
                        clear_terminal()
                        print("No input detected, exiting...")
                        return

                    # check if new movie id is non-negative int
                    if not runtime_y.isdigit() or runtime_y.isdigit() < 0:
                        clear_terminal()
                        print("Invalid input, must be non-negative integer, exiting...")
                        return

                    # check if x <= y
                    if not (runtime_x <= runtime_y):
                        print("Error: x is not less than or equal to y")
                        return

                    runtime_query = "runtime BETWEEN '" + runtime_x + "' AND '" + runtime_y + "'"
                    # release_date BETWEEN 'date_x' AND 'date_y'
            else:
                operator = ""
                # does nothing


        #vote_average
        if user == "3":
            vote_average_x = ""
            vote_average_y = ""
            operator = ""
            print("1: vote average > x")
            print("2: vote average < x ")
            print("3: vote average between x and y (x <= y)")
            print("4: turn filter off")
            print("5: leave filter as is")

            vote_average_user = input("input option (default 5):")

            # pick operator
            if vote_average_user == "1":
                operator = ">="
            if vote_average_user == "2":
                operator = "<="
            
            if vote_average_user == "4":
                vote_average_query = ""
            elif vote_average_user == "5":
                #placeholder, does nothing and exits
                operator = ""
            elif vote_average_user == "3":
                vote_average_x = input("Input vote_average x (float between 0-10): ")

                # check if rating is float or not
                try: 
                    vote_average_x = float(vote_average_x)
                except:
                    print("Invalid input, exiting...")
                    return

                #check if between 0-10
                if not (0 <= vote_average_x <= 10):
                    print("Invalid input, exiting...")
                    return

                # make query for options 1 and 2
                if vote_average_user == "1" or vote_average_user == "2":
                    vote_average_query = "vote_average " + operator + " '" + str(vote_average_x) + "'"
                    # runtime <=/>= 'runtime_x'
                
                
                # release date between x and y
                # get second date
                if vote_average_user == "3":
                    vote_average_y = input("Input runtime y (on-negative int): ")
                    
                    # check if rating is float or not
                    try: 
                        vote_average_y = float(vote_average_y)
                    except:
                        print("Invalid input, exiting...")
                        return

                    # check if between 0-10
                    if not (0 <= vote_average_y <= 10):
                        print("Invalid input, exiting...")
                        return

                    # check if x <= y
                    if not (vote_average_x <= vote_average_y):
                        print("Error: x is not less than or equal to y")
                        return

                    vote_average_query = "vote_average BETWEEN '" + vote_average_y + "' AND '" + vote_average_y + "'"
                    # release_date BETWEEN 'date_x' AND 'date_y'

                else:
                    operator = ""
                    # does nothing

        #vote count
        if user == "4":
            vote_count_x = ""
            vote_count_y = ""
            operator = ""
            print("1: vote count > x")
            print("2: vote count < x ")
            print("3: vote count between x and y (x <= y)")
            print("4: turn filter off")
            print("5: leave filter as is")

            vote_count_user = input("input option (default 5):")

            # pick operator
            if vote_count_user == "1":
                operator = ">="
            if vote_count_user == "2":
                operator = "<="
            
            if vote_count_user == "4":
                vote_count_query = ""
            elif vote_count_user == "5":
                #placeholder, does nothing and exits
                operator = ""
            elif vote_count_user == "3":
                vote_count_x = input("Input runtime x (non-negative int): ")

                # check if int and non-negative
                if not vote_count_x.strip():
                    clear_terminal()
                    print("No input detected, exiting...")
                    return

                # check if new movie id is non-negative int
                if not vote_count_x.isdigit() or vote_count_x.isdigit() < 0:
                    clear_terminal()
                    print("Invalid input, must be non-negative integer, exiting...")
                    return

                # make query for options 1 and 2
                if vote_count_user == "1" or vote_count_user == "2":
                    vote_count_query = "vote_count " + operator + " '" + vote_count_x + "'"
                    # runtime <=/>= 'runtime_x'
                
                
                # release date between x and y
                # get second date
                if vote_count_user == "3":
                    vote_count_y = input("Input runtime y (on-negative int): ")
                    
                    # check if int and non-negative
                    if not vote_count_y.strip():
                        clear_terminal()
                        print("No input detected, exiting...")
                        return

                    # check if new movie id is non-negative int
                    if not vote_count_y.isdigit() or vote_count_y.isdigit() < 0:
                        clear_terminal()
                        print("Invalid input, must be non-negative integer, exiting...")
                        return

                    # check if x <= y
                    if not (vote_count_x <= vote_count_y):
                        print("Error: x is not less than or equal to y")
                        return

                    vote_count_query = "vote_count BETWEEN '" + vote_count_x + "' AND '" + vote_count_y + "'"
                    # release_date BETWEEN 'date_x' AND 'date_y'

                else:
                    operator = ""
                    # does nothing

        if user == "5":

            # subquery processing
            # if filters 
            subquery = [release_date_query, runtime_query, vote_average_query, vote_count_query]
            subquery = [item for item in subquery if item] # filter out empty strings
            if not subquery:
                print("No filters entered, exiting")
                return
            subquery = " AND ".join(subquery)

            print("subquery:" + subquery)
            

            main_query = '''
                SELECT title, movie_id
                FROM movie
                WHERE ''' + subquery + ''';
            '''
            #print(main_query)
            #input("press enter to continue")


            # run query
            c.execute(main_query)
            result = c.fetchall()
            paginate(result,30)
            #return
            
        if user == "6":
            return

            
# -----------------------------------
# Backup R14 Functions
# -----------------------------------     


def search_credit():
    search = input("Input credited person name search:") #user input

    # query
    query = '''
        SELECT name, person_id 
        FROM credit
        WHERE name like %s 
        LIMIT 10
    '''

    # run and print query
    c.execute(query, ('%' + search + '%',))
    result = c.fetchall()
    print_table(result)

    person_id = input("Enter person_id you'd like to get information from:")

    if not person_id.isdigit() or person_id.isdigit() < 0 or person_id.strip() == "":
        clear_terminal()
        print("Invalid input, must be non-negative integer, exiting...")
        return
    else:
        return person_id


def productions_affiliations(person_id):
    print("Selecting the productions that given person has worked with in...")
    query = '''
        SELECT p.production_name as "Production Studio", COUNT(*) AS "Times featured"
        FROM credit c
        JOIN comprises_of co ON c.person_id = co.person_id
        JOIN movie m ON co.movie_id = m.movie_id
        JOIN produced_by pb ON m.movie_id = pb.movie_id
        JOIN production p ON pb.production_id = p.production_id
        WHERE c.person_id = %s
        GROUP BY p.production_name
        ORDER BY movie_count Desc;
    '''
    
    c.execute(query, (person_id,))
    result = c.fetchall()
    print_table(result)



def search_production():
    search = input("Input production studio name search:") #user input

    # query
    query = '''
        SELECT production_name, production_id 
        FROM production
        WHERE production_name like %s 
        LIMIT 10
    '''

    # run and print query
    c.execute(query, ('%' + search + '%',))
    result = c.fetchall()
    print_table(result)

    production_id = input("Enter production_id you'd like to get information from:")

    if not production_id.isdigit() or production_id.isdigit() < 0 or production_id.strip() == "":
        clear_terminal()
        print("Invalid input, must be non-negative integer, exiting...")
        return
    else:
        return production_id


def most_hired_per_production(production_id):
    print("Selecting the people who have collaborated most with the given production studio...")
    query = '''
        SELECT c.person_id, c.name, COUNT(*) AS "Collaboration Count"
        FROM credit c
        INNER JOIN comprises_of co ON c.person_id = co.person_id
        INNER JOIN movie m ON co.movie_id = m.movie_id
        INNER JOIN produced_by pb ON m.movie_id = pb.movie_id
        INNER JOIN production p ON pb.production_id = p.production_id
        WHERE p.production_id = %s
        GROUP BY c.person_id, c.name
        ORDER BY collaborations DESC;
    '''

    c.execute(query, (production_id,))
    result = c.fetchall()
    print_table(result)


def actor_genre_count(person_id):
    print("Selecting the genres that given person has participated in...")
    query = '''
        WITH PersonMovies AS (
        SELECT m.movie_id
        FROM movie m
        INNER JOIN comprises_of co ON m.movie_id = co.movie_id
        WHERE co.person_id = %s
        )
        SELECT g.genre_name, COUNT(*) AS movie_count
        FROM genre g
        INNER JOIN classified_in ci ON g.genre_id = ci.genre_id
        INNER JOIN PersonMovies pm ON ci.movie_id = pm.movie_id
        GROUP BY g.genre_name
        ORDER BY movie_count DESC;
    '''

    c.execute(query, (person_id,))
    result = c.fetchall()
    print_table(result)

def R14():
    while True:
        print("1: credited worker affiliations")
        print("2: search most hired people per production studio")
        print("3: credited person's favourite genres")
        print("4: exit")

        user = input("Select option:")

        if user == "1":
            productions_affiliations(search_credit())
        elif user == "2":
            most_hired_per_production(search_production())
        elif user == "3":
            actor_genre_count(search_credit())
        elif user == "4":
            return

# -----------------------------------
# R15 Functions
# -----------------------------------

def add_rating():
    movie_id = input("Enter the Movie ID you want to rate: ")
    query = '''
        SELECT vote_average, vote_count, title
        FROM movie 
        WHERE movie_id = %s
        '''
    c.execute(query, (movie_id,))
    result = c.fetchone()

    # quit if no movie found
    if not result:
        print("Movie not found, exiting...")
        return
    
    vote_average = float(result[0])
    vote_count = int(result[1])
    title = result[2]

    print("Movie found: " + title)
    rating = input("Enter the rating you would like to add (between 0-10, inclusive, can be float):")

    # check if rating is integer or not
    try: 
        rating = float(rating)
    except:
        print("Invalid input, exiting...")
        return

    if not (0 <= rating <= 10):
        print("Invalid input, exiting...")
        return

    vote_average = (vote_average * vote_count + rating) / (vote_count + 1) #calculate new vote_average
    vote_average = str(round(vote_average, 3)) #round to 2 decimal places and convet to string
    vote_count += 1 #increment vote_count
    vote_count = str(vote_count) # convert to string


    update_query = '''
        UPDATE movie
        SET vote_average = %s, vote_count = %s
        WHERE movie_id = %s
    '''

    c.execute(update_query, (vote_average, vote_count, movie_id))
    netflixdb.commit()

    print("new average vote: " + str(vote_average))
    print("new vote_count: " + str(vote_count))

# -----------------------------------
# R16 Functions
# -----------------------------------

def create_indexes():

    # checks if indexes are already created. if so, don't run create index functions
    check_query = '''
        SHOW INDEX
        FROM movie
        WHERE Key_name = "idx_movie_title"
    '''

    index_queries = '''
        CREATE INDEX idx_movie_title ON movie(title);
        CREATE INDEX idx_classified_in_genre_id ON classified_in(genre_id);
        CREATE INDEX idx_classified_in_movie_id ON classified_in(movie_id);
        CREATE INDEX idx_produced_by_movie_id ON produced_by(movie_id);
        CREATE INDEX idx_produced_by_production_id ON produced_by(production_id);
        CREATE INDEX idx_production_name ON production(production_name);
        CREATE INDEX idx_genre_name ON genre(genre_name);
        CREATE INDEX idx_crew_person_id ON crew(person_id);
    '''

    c.execute(check_query)
    result = c.fetchone()

    if not result:
        c.execute(index_queries)


# -----------------------------------
# Connecting to DB
# -----------------------------------

# check if credentials are saved to log into db

# init variables
save = ""
username = ""
creds_saved = ""

# read credentials files
with open(cwd + "\\scripts\\creds\\user.txt") as f:
    username = f.read()

with open(cwd + "\\scripts\\creds\\pw.txt") as f:
    pw = f.read()


# check if creds files are empty:
# if either are empty, prompt user to log in.
# else, prompt user if they want to log in with previous credentials
if len(username) == 0 or len(pw) == 0:
    creds_saved = False
    print("No saved credentials found")
else:
    save = input("Found saved credentials, load? ([y]/n): ") 
    if save == "n":
        creds_saved = False
    else:
        creds_saved = True

# no credentials found, prompt user to log in (and if they want to save credentials)
if not creds_saved:
    username = input("Enter DB Username: ")
    pw = getpass(prompt="Enter DB Password: ")
    print("Do you want to save these credentials for later?")
    save = input("WARNING!! UNSECURE TO SAVE CREDS!! (y/[n]): ")
    if save == "y":
        with open(cwd + "\\scripts\\creds\\user.txt", 'w') as f:
            f.write(username)

        with open(cwd + "\\scripts\\creds\\pw.txt", 'w') as f:
            f.write(pw)

# connect to sql
netflixdb = mysql.connector.connect(
    host="localhost",
    user=username,
    password=pw,
    allow_local_infile=True,
)

# connect cursor to db to execute queries
c = netflixdb.cursor()
redo_database() #create tables and load tables with csvs

# -----------------------------------
# Menu Nav
# -----------------------------------

# these functions are for individual sub-menus
# each involve options to run specific functions, related to the sub menu
# some sub-menus are specifically the featured function, whilst one (stats) contains
# multiple features related to statistics


def R6():
    clear_terminal()
    while True:
        print("1: Search titles, fuzzy")
        print("2: Search titles with genre")
        print("3: Search featuring person, fuzzy")
        print("4: Basic details and synopsis (from movie ID)")
        print("5 (R12): Search Metadata (from movie ID)")
        print("6 (R13): Advanced Search filters")
        print("7 (R15): Rate a Movie!")
        print("8: Exit to main menu")
        user = input("Select option (default 7): ")
        if user == "1":
            search_title()
        elif user == "2":
            search_genre()
        elif user == "3":
            search_credited_person()
        elif user == "4":
            movie_data_simple()
        elif user == "5":
            move_data_full()
        elif user == "6":
            advanced_search()
        elif user == "7":
            add_rating()
        else:
            return

def R7():
    clear_terminal()
    while True:
        print("1: Search movies produced by production, fuzzy")
        print("2: Exit to main menu")
        user = input("Select option (default 2): ")
        if user == "1":
            search_production()
        else:
            return

def R8():
    clear_terminal()
    while True:
        print("1: Update movie attributes")
        print("2: Insert New Movie")
        print("3: Delete Movie")
        print("2: Exit to main menu")
        user = input("Select option (default 2): ")

        if user == "1":
            update_movie()
        elif user == "2":
            insert_movie()
        elif user == "3":
            delete_movie()
        else:
            return

def Stats():
    clear_terminal()
    while True:
        print("1 (R11): Popular Genres")
        print("2 (R11): Most successful Genres")
        print("3 (R9): Average rating per crew member")
        print("4 (R10): Best production studios per genre")
        print("5: (R14) : Stats about people")
        print("6: Exit to previous menu")
        user = input("Select option: ")

        if user == "1":
            genre_count()
        elif user == "2":
            best_genre()
        elif user == "3":
            success_rate_per_crew()
        elif user == "4":
            best_production_studios()
        elif user == "5":
            R14()
        elif user == "6":
            clear_terminal()
            return



# main menu
def menu():
    while True:
        clear_terminal()
        print("1 (R6 + R15): Searching and metadata")
        print("2 (R7): search productions")
        print("3 (R8): Edit Movie Details")
        print("4 (R9, R10, R11, R14): Get statistics")
        print("5: Remake database")
        print("6: Exit")
        user = input("Select option: ")
        if user == "1":
            R6()
        elif user == "2":
            R7()
        elif user == "3":
            R8()
        elif user == "4":
            Stats()
        elif user == "5":
            redo_database()
        elif user == "6":
            break

c.execute("Use Netflix")
create_indexes() #create indexes R16
menu()
netflixdb.close()