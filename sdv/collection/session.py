"""Connection and ownership adapter; the research collector remains unchanged."""
import os


def connect_external():
    import carla
    host = os.environ.get('SDV_CARLA_HOST', '127.0.0.1')
    port = int(os.environ.get('SDV_CARLA_PORT', '2000'))
    client = carla.Client(host, port, worker_threads=2)
    client.set_timeout(10.0)
    version = client.get_server_version()
    if version != client.get_client_version():
        raise RuntimeError(f'CARLA 버전 불일치: 서버 {version}, Python {client.get_client_version()}')
    return client, client.get_world()


class CollectionSession:
    """Tracks only actors spawned by this job and restores world settings."""
    def __init__(self, client, traffic_manager, server_job=None):
        self.client = client
        self.traffic_manager = traffic_manager
        self.server_job = server_job
        self.actor_ids = set()
        self.world = None
        self.settings = None
        self.weather = None

    def begin_world(self, world):
        self.cleanup_world()
        self.world = world
        self.settings = world.get_settings()
        self.weather = world.get_weather()

    def own(self, actor_ids):
        self.actor_ids.update(actor_ids)

    def cleanup_world(self):
        from sdv.runtime.actors import destroy_owned_actors, _cleanup_note
        if self.world is None: return
        try:
            self.client.set_timeout(1.0)
            actors = list(self.world.get_actors(list(self.actor_ids))) if self.actor_ids else []
            for actor in actors:
                if actor.type_id == 'controller.ai.walker':
                    try: actor.stop()
                    except RuntimeError: pass
            destroy_owned_actors(self.client, actors, self.server_job, seconds=3)
            if self.settings is not None: self.world.apply_settings(self.settings)
            if self.weather is not None: self.world.set_weather(self.weather)
        except Exception as error:
            _cleanup_note({'collection_cleanup_error': str(error)})
        finally:
            self.actor_ids.clear()
            self.world = self.settings = self.weather = None

    def close(self):
        self.cleanup_world()
        try: self.traffic_manager.set_synchronous_mode(False)
        except RuntimeError: pass
