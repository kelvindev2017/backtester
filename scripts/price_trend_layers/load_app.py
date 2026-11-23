import sys, logging
sys.path.insert(0, "../..")
logging.disable(logging.WARNING)
src = open("../../App.py").read()
src = src[:src.index("\nif check_password():")]
G = {"__name__": "app_lib", "__file__": "../../App.py"}
exec(compile(src, "App.py", "exec"), G)
