# tracked_array.py
'''
定义一个可追踪操作的数组代理类，支持嵌套结构。
'''

class TrackedArray:
    def __init__(self, array, _history=None, _path=()):
        self.history = _history if _history is not None else []
        
        self._path = _path
        
        self._array = []
        for i, item in enumerate(array):
            if isinstance(item, list):
                self._array.append(TrackedArray(item, _history=self.history, _path=self._path + (i,)))
            else:
                self._array.append(item)
        if not self._path:
            self.history.append({"array": self._get_raw_copy(), "highlights": {}})

    def _get_raw_copy(self):
        raw_copy = []
        for item in self._array:
            if isinstance(item, TrackedArray):
                raw_copy.append(item._get_raw_copy())
            else:
                raw_copy.append(item)
        return raw_copy

    def __len__(self):
        return len(self._array)

    def __getitem__(self, key):
        item = self._array[key]
        if not isinstance(item, TrackedArray):
            resolved_key = self._resolve_key(key)
            highlights = {"comparing": {self._path + (resolved_key,)}}
            self.history.append({"array": self._get_raw_copy(), "highlights": highlights})
        return item

    def __setitem__(self, key, value):
        resolved_key = self._resolve_key(key)
        if isinstance(value, list):
            new_path = self._path + (resolved_key,)
            self._array[key] = TrackedArray(value, _history=self.history, _path=new_path)
        else:
            self._array[key] = value
        highlights = {"swapping": {self._path + (resolved_key,)}}
        self.history.append({"array": self._get_raw_copy(), "highlights": highlights})

    def __repr__(self):
        return repr(self._get_raw_copy())

    def _resolve_key(self, key):
        if isinstance(key, int):
            if key < 0:
                return len(self._array) + key
            return key
        return -1

    def append(self, value):
        new_index = len(self._array)
        if isinstance(value, list):
            new_path = self._path + (new_index,)
            self._array.append(TrackedArray(value, _history=self.history, _path=new_path))
        else:
            self._array.append(value)
        
        highlights = {"changing": {self._path + (new_index,)}}
        self.history.append({"array": self._get_raw_copy(), "highlights": highlights})
