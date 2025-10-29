# main.py
'''
用户交互的主程序。
能够自动发现sort.py中的排序算法，并提供一个暗色调的设置界面。
'''
import pygame
import pygame_gui
import random
import inspect
import sort as sort_module # 将整个sort模块导入，以便我们通过名字调用函数
import math
from time import time
from visualizer import Visualizer
from tracked_array import TrackedArray

def get_sorting_algorithms():
    """
    使用inspect模块自动发现sort.py中所有以大写字母开头的排序函数。
    返回一个字典，键是函数名（字符串），值是函数对象本身。
    """
    algorithms = {}
    for name, func in inspect.getmembers(sort_module, inspect.isfunction):
        # 我们的约定：排序算法函数名以大写字母开头
        if name[0].isupper():
            algorithms[name] = func
    return algorithms

def generate_random_array(n, min_val, max_val):
    global last_input, last_output, last_seed
    if last_input != [n, min_val, max_val] or not last_output:
        if max_val == -1:
            last_output = list(range(1, n + 1))
            random.shuffle(last_output)
        else:
            last_output = [random.randint(min_val, max_val) for _ in range(n)]
        last_seed = math.sin(time())
    elif not last_seed:
        last_seed = math.sin(time())
    last_input = [n, min_val, max_val]
    return last_output

def parse_user_array(text):
    try:
        if not text: return None
        return [int(x.strip()) for x in text.split(',') if x.strip()]
    except ValueError:
        return None

def setup_screen(width=800, height=600, algorithms=None):
    global last_function
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Sorting Algorithm Visualizer - Settings")
    # manager = pygame_gui.UIManager((width, height), 'theme.json')
    manager = pygame_gui.UIManager((width, height))
    clock = pygame.time.Clock()

    if not algorithms:
        algorithms = {}

    algo_names = list(algorithms.keys())

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, 20), (230, 30)), text="1. Select Sorting Algorithm:", manager=manager)
    algo_dropdown = pygame_gui.elements.UIDropDownMenu(
        options_list=algo_names,
        # starting_option=algo_names[0] if algo_names else "",
        starting_option=last_function if last_function else (algo_names[0] if algo_names else ""),
        relative_rect=pygame.Rect((70, 50), (280, 40)),
        manager=manager
    )

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, 110), (280, 30)), text="2. Choose Array Generation Method:", manager=manager)

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((70, 140), (350, 30)), text="Manual Input (comma-separated, e.g., 5,2,8,1):", manager=manager)
    array_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((70, 170), (660, 40)), manager=manager)

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((70, 230), (700, 30)), text="Or Randomly Generate (the generated array won't change if the parameters don't gain changed):", manager=manager)
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((90, 260), (80, 30)), text="Size:", manager=manager)
    size_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((170, 260), (100, 30)), manager=manager)
    size_input.set_text(str(last_input[0]))

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((290, 260), (80, 30)), text="Min Value:", manager=manager)
    min_val_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((370, 260), (100, 30)), manager=manager)
    min_val_input.set_text(str(last_input[1]))

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((490, 260), (80, 30)), text="Max Value:", manager=manager)
    max_val_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((570, 260), (100, 30)), manager=manager)
    max_val_input.set_text(str(last_input[2]))

    start_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((width // 2 - 100, 340), (200, 50)), text='Start Visualization', manager=manager)
    message_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, 400), (700, 40)), text="", manager=manager)
    message_label.set_text_alpha(0)

    while True:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == start_button:
                    arr = None
                    message_label.set_text_alpha(255)
                    user_array_text = array_input.get_text()
                    if user_array_text:
                        arr = parse_user_array(user_array_text)
                        if arr is None:
                            message_label.set_text("Error: Invalid format for manual array. Please check.")
                            continue
                        if len(arr) < 2:
                            message_label.set_text("Error: Array must contain at least two elements.")
                            continue
                    else:
                        try:
                            n = int(size_input.get_text())
                            min_val = int(min_val_input.get_text())
                            max_val = int(max_val_input.get_text())
                            if n < 2 or (min_val >= max_val and not max_val == -1) or n > 100000:
                                message_label.set_text("Error: Invalid parameters (Size > 1, Min < Max, Size <= 100000).")
                                continue
                            arr = generate_random_array(n, min_val, max_val)
                        except (ValueError, TypeError):
                            message_label.set_text("Error: Parameters for random generation must be valid integers.")
                            continue

                    if not algo_names:
                        message_label.set_text("Error: No sorting algorithms found in sort.py.")
                        continue

                    selected_algo_name = algo_dropdown.selected_option[0]
                    sort_function = algorithms[selected_algo_name]
                    last_function = selected_algo_name

                    message_label.set_text(f"Array generated. Size: {len(arr)}. Algorithm: {selected_algo_name}.")
                    manager.draw_ui(screen)
                    pygame.display.flip()
                    pygame.time.wait(1000)

                    return sort_function, arr

            manager.process_events(event)

        manager.update(time_delta)
        screen.fill(pygame.Color('#282c34'))
        manager.draw_ui(screen)
        pygame.display.flip()

def main():
    global last_input, last_output, last_function, last_seed
    # 程序启动时，自动发现所有可用的排序算法
    available_algorithms = get_sorting_algorithms()
    print("Discovered sorting algorithms:", list(available_algorithms.keys()))
    
    last_seed = None
    last_input = [100, 1, 100]
    last_output = []
    last_function = None

    while True:
        sort_function, array_to_sort = setup_screen(algorithms=available_algorithms)

        if sort_function is None or array_to_sort is None:
            print("Program exited.")
            break

        print(f"Executing '{sort_function.__name__}' and recording history...")
        tracked_arr = TrackedArray(array_to_sort)

        # 智能调用函数，处理不同参数签名
        sig = inspect.signature(sort_function)
        cur_time = time()
        if len(sig.parameters) == 3: # 像QuickSort(arr, low, high)
            sort_function(tracked_arr, 0, len(tracked_arr) - 1)
        elif len(sig.parameters) == 2 and last_seed: # 像MonkeySort(arr, seeds)
            sort_function(tracked_arr, last_seed)
        else: # 像BubbleSort(arr)
            sort_function(tracked_arr)
        
        cur_time = time() - cur_time

        print(f"Recording complete. Used {cur_time:.6f} s and captured {len(tracked_arr.history)} steps.")

        visualizer = Visualizer()
        result = visualizer.run(tracked_arr.history)

        if result == 'quit':
            print("Program exited from visualizer.")
            break

    pygame.quit()

if __name__ == "__main__":
    main()
