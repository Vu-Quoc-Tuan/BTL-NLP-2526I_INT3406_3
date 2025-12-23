from typing import Dict, Any, Optional

class Registry:
    """
    Lớp Registry đơn giản để đăng ký và truy xuất các module/class.
    Rất hữu ích để quản lý các kiến trúc model, optimizer, scheduler khác nhau
    mà không cần sửa code cứng (hard-code).
    """
    
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}
        
    def register(self, name: Optional[str] = None):
        """
        Decorator để đăng ký một class hoặc function.
        Cách dùng:
            @registry.register()
            class MyModel...
            
            @registry.register("alias")
            class MyModel...
        """
        def _register(obj):
            module_name = name if name else obj.__name__
            if module_name in self._registry:
                raise KeyError(f"{module_name} is already registered in {self._name} registry")
            self._registry[module_name] = obj
            return obj
        return _register
    
    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name} registry. Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def __repr__(self):
        return f"<Registry {self._name}: {list(self._registry.keys())}>"
