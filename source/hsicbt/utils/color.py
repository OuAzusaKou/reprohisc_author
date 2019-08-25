

TEXT_GREEN = "\033[1;32;40m"
TEXT_YELLOW = "\033[1;33;40m"
TEXT_BLACK = "\033[0;37;48m"



def print_highlight(msg, ctype='yellow'):

	if ctype == 'yellow':
		color = TEXT_YELLOW
	elif ctype == 'green':
		color = TEXT_GREEN

	print("{}{}{}".format(color, msg, TEXT_BLACK))
