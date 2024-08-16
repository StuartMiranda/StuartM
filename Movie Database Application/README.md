# CS338 - Group 18 

Through this project, we are aiming to work with a sample database to simulate the MovieLens dataset using MySQL on the school server. The database contains data on the movies released on or before July 2017. Users will be able to query, add, delete, and modify data through a Python-based command-line interface (CLI).

# Dependencies

In order to use this project effectively, the user must have:
1. MySQL Server 8.4 installed (THIS APPLICATION WILL NOT WORK WITH VERSIONS OTHER THAN 8.4)
2. Any version of Python 3 installed on their machine

 a. the environmental variables for python and its scripts properly configured (when installing python, there is an option to add python to PATH. This is 100% necessary) 
3. The following modules. These can be installed via a script in `main.bat`, or you can use the pip command.

 a. mysql-connector-python

 b. getpass4

 c. tabulate

 d. sqlalchemy

 e. pandas

 f. pymysql

# Usage
Simply copy this repository and run `main.bat`
- it will prompt you to install some python modules which will be required
- prompt you to select "CLI" or "GUI"

## CLI
When selecting the CLI option in main.bat, it will prompt you to:
- enter DB crednetials, and save if preferred
- prompt to recreate the DB with prod/sample dataset
- present the user to a menu with different options

![image](https://github.com/palakagarwal6/CS338/assets/170012515/72c1e590-35e4-4bda-8a91-f523cc2c9865)


## GUI
## IMPORTANT STEPS FOR GUI
Before running the GUI: 

navigate to `CS338-main\GUI\db.py` and edit line 23 to enter your db connection password (replace [your_password])
![image](https://github.com/user-attachments/assets/d70fb3f4-34f3-4bf6-9331-5f8a304d70a7)

navigate to `CS338-main\GUI\gui_main.py` and edit line 202 to enter your db connection password (replace [your_password])
![image](https://github.com/user-attachments/assets/93cda024-bd07-4662-9a7c-4a8e2e05138b)

When selecting the GUI option in main.bit, it will:
- open a browser to a local connection GUI
- in the left hand bar presents the option to load the prod/sample database, as well as some additional menu items.

![image](https://github.com/user-attachments/assets/43766bb7-6f5f-44cd-a1ea-12937dddac3d)
