from typing import Any, Type, Dict
from service.interface import (
    ITableLocateService,
    IOCRService,
    ITableStructureService,
    ICustomService,
)
from service.det import PaddleLocateService, YoloLocateService
from service.rec import PaddleOCRService
from service.structure import TableTransformerService, CycleCenterNetService
from service.custom.ScoreEvaluation_v2 import A4ScoreEvaluation


class ServiceRegistry:
    """服务注册中心，统一管理服务实例"""

    def __init__(self):
        self._services: Dict[Type, Dict[str, Any]] = {}

    def register(self, service_type: Type, service: Any, name: str = "default"):
        """注册服务"""
        if service_type not in self._services:
            self._services[service_type] = {}
        self._services[service_type][name] = service

    def get(self, service_type: Type, name: str = "default") -> Any:
        """获取服务实例"""
        services_of_type = self._services.get(service_type)
        if not services_of_type:
            raise ValueError(f"Service type {service_type.__name__} not registered!")
        service = services_of_type.get(name)
        if not service:
            raise ValueError(
                f"Service {service_type.__name__} with name '{name}' not registered!"
            )
        return service


class DefaultServiceRegistry(ServiceRegistry):
    """预注册常用服务实现的注册中心"""

    def __init__(self, overrides: Dict[Type, Any] = None):
        super().__init__()
        self.register(ITableLocateService, PaddleLocateService(), name="paddle")
        self.register(
            ITableStructureService,
            TableTransformerService(),
            name="table_transfromer",
        )
        self.register(IOCRService, PaddleOCRService(), "paddle")

        # 覆盖用户指定的服务
        if overrides:
            for service_type, (name, service) in overrides.items():
                self.register(service_type, service, name=name)


registry = DefaultServiceRegistry()  # 默认配置
registry.register(ITableLocateService, YoloLocateService(), "yolo")  # 按需覆盖
registry.register(
    ITableStructureService, CycleCenterNetService(), "cyclecenter_net"
)  # 按需覆盖
# registry.register(ICustomService, A4ScoreEvaluation, name="a4table_score_eval")
