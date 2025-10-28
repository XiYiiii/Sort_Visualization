# visualizer.py
'''
负责将排序过程可视化的模块。
使用 pygame 和 pygame_gui 实现交互式暗色调界面和播放控制。
'''
import pygame
import pygame_gui

class Visualizer:
    def __init__(self, width=1200, height=750):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Sorting Algorithm Visualizer")

        self.ui_manager = pygame_gui.UIManager((width, height))

        self.BACKGROUND_COLOR = '#282c34'
        self.BAR_COLOR = '#61afef'
        self.COMPARING_COLOR = '#e5c07b'
        self.SWAPPING_COLOR = '#98c379'
        self.SORTED_COLOR = '#c678dd'

        self.clock = pygame.time.Clock()
        self._build_ui()

    def _build_ui(self):
        ui_height = self.height - 50
        self.progress_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((20, ui_height - 30), (self.width - 40, 20)),
            start_value=0.0, value_range=(0.0, 1.0), manager=self.ui_manager
        )
        self.play_pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, ui_height), (80, 30)),
            text='Pause', manager=self.ui_manager
        )
        self.speed_buttons = {
            '0.5x': pygame_gui.elements.UIButton(relative_rect=pygame.Rect((110, ui_height), (50, 30)), text='0.5x', manager=self.ui_manager),
            '1x': pygame_gui.elements.UIButton(relative_rect=pygame.Rect((170, ui_height), (50, 30)), text='1x', manager=self.ui_manager),
            '2x': pygame_gui.elements.UIButton(relative_rect=pygame.Rect((230, ui_height), (50, 30)), text='2x', manager=self.ui_manager),
            '4x': pygame_gui.elements.UIButton(relative_rect=pygame.Rect((290, ui_height), (50, 30)), text='4x', manager=self.ui_manager),
        }
        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((360, ui_height), (70, 30)), text="Custom:", manager=self.ui_manager)
        self.speed_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((430, ui_height), (80, 30)), manager=self.ui_manager
        )
        self.speed_input.set_text('1.0')
        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((520, ui_height), (60, 30)),
            text='1.0x', manager=self.ui_manager
        )
        self.back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.width - 120, ui_height), (100, 30)),
            text='Back to Menu', manager=self.ui_manager
        )

    def draw_bars(self, data, sorted_indices):
        draw_surface_height = self.height - 100
        draw_surface = pygame.Surface((self.width, draw_surface_height))
        draw_surface.fill(pygame.Color(self.BACKGROUND_COLOR))

        array = data.get("array", [])
        highlights = data.get("highlights", {})

        if not array:
            self.screen.blit(draw_surface, (0, 0))
            return

        bar_width = self.width / len(array)
        max_val = max(array) if array else 1
        unit_height = draw_surface_height * 0.95 / max_val

        for i, val in enumerate(array):
            x_pos = i * bar_width
            bar_height = val * unit_height
            y_pos = draw_surface_height - bar_height

            color = self.BAR_COLOR
            path = (i,)
            if i in sorted_indices:
                color = self.SORTED_COLOR
            elif "swapping" in highlights and path in highlights["swapping"]:
                color = self.SWAPPING_COLOR
            elif "comparing" in highlights and path in highlights["comparing"]:
                color = self.COMPARING_COLOR

            pygame.draw.rect(draw_surface, pygame.Color(color), (x_pos, y_pos, bar_width, bar_height))
            pygame.draw.rect(draw_surface, pygame.Color(self.BACKGROUND_COLOR), (x_pos, y_pos, bar_width, bar_height), 1)
        self.screen.blit(draw_surface, (0, 0))


    def run(self, history):
        running = True
        paused = True
        self.play_pause_button.set_text('Play')
        speed_multiplier = 1.0
        base_fps = 60

        history_index = 0
        max_index = len(history) - 1 if history else 0
        self.progress_slider.value_range = (0.0, float(max_index))
        if self.progress_slider.value_range[1] > self.progress_slider.value_range[0]:
             self.progress_slider.rebuild()

        sorted_indices = set()
        is_finished = False

        while running:
            time_delta = self.clock.tick(base_fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.play_pause_button:
                        paused = not paused
                        self.play_pause_button.set_text('Play' if paused else 'Pause')
                        if is_finished and not paused:
                            is_finished = False
                            history_index = 0
                            sorted_indices.clear()
                    elif event.ui_element == self.back_button:
                        return 'back'
                    else:
                        for speed_text, button in self.speed_buttons.items():
                            if event.ui_element == button:
                                speed_multiplier = float(speed_text.replace('x', ''))
                                self.speed_label.set_text(f"{speed_multiplier:.1f}x")
                                self.speed_input.set_text(str(speed_multiplier))

                if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == self.speed_input:
                    try:
                        custom_speed = float(event.text)
                        speed_multiplier = max(0.1, min(100.0, custom_speed))
                        self.speed_label.set_text(f"{speed_multiplier:.1f}x")
                        self.speed_input.set_text(str(speed_multiplier))
                    except ValueError:
                        self.speed_input.set_text(str(speed_multiplier))

                if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if event.ui_element == self.progress_slider:
                        history_index = int(event.value)
                        is_finished = False
                        sorted_indices.clear()

                self.ui_manager.process_events(event)

            self.ui_manager.update(time_delta)

            if not paused and not is_finished:
                steps_to_advance = max(1, int(time_delta * base_fps * speed_multiplier))
                history_index += steps_to_advance
                if history_index >= max_index:
                    history_index = max_index
                    is_finished = True
                    paused = True
                    self.play_pause_button.set_text('Replay')

            self.screen.fill(pygame.Color(self.BACKGROUND_COLOR))

            current_state = history[history_index] if history else {"array": [], "highlights": {}}
            if is_finished:
                final_array = history[-1]["array"]
                self.draw_bars({"array": final_array, "highlights": {}}, set(range(len(final_array))))
            else:
                self.draw_bars(current_state, sorted_indices)

            if not self.progress_slider.is_focused:
                 self.progress_slider.set_current_value(history_index)
            self.ui_manager.draw_ui(self.screen)

            pygame.display.flip()

        return 'quit'
