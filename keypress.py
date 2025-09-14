import curses

def test_keys(stdscr):
    stdscr.addstr(0, 0, "Press HOME, END, Ctrl+arrows, then 'q' to quit")
    while True:
        key = stdscr.getch()
        if key == ord('q'):
            break
        stdscr.addstr(1, 0, f"Key pressed: {key} (0x{key:x})          ")
        stdscr.refresh()

curses.wrapper(test_keys)
