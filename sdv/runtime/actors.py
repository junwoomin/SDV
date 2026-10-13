"""Ownership-aware, bounded sensor and actor cleanup."""
import time
from sdv.paths import RUNTIME

def _cleanup_note(record):
    import json
    try:
        path = RUNTIME / 'logs/actor-cleanup.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass  # Cleanup diagnostics must never replace the original exception.

def destroy_owned_actors(client, actors, job=None, seconds=1):
    """Bound RPC cleanup; never contact a replacement or already stopped server."""
    report = {'actors': 0, 'reason': '', 'errors': []}
    if job is not None and (job.stopped or job.poll() is not None):
        report['reason'] = 'server_not_alive'
        _cleanup_note(report)
        return report
    unique = {actor.id: actor for actor in actors if actor is not None}
    ordered = sorted(unique.values(), key=lambda actor: not actor.type_id.startswith('sensor.'))
    report['actors'] = len(ordered)
    if not ordered: return report
    deadline = time.monotonic() + seconds
    try:
        client.set_timeout(.25)
        for actor in ordered:
            if time.monotonic() >= deadline:
                report['reason'] = 'cleanup_deadline'
                break
            if job is not None and (job.stopped or job.poll() is not None):
                report['reason'] = 'server_not_alive'
                break
            if actor.type_id.startswith('sensor.'):
                try:
                    if actor.is_listening: actor.stop()
                except Exception as error:
                    report['errors'].append(str(error))
        if not report['reason'] and time.monotonic() < deadline:
            import carla
            responses = client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in ordered], False)
            report['errors'].extend(r.error for r in responses if r.error)
    except Exception as error:
        report['errors'].append(str(error))
    finally:
        try: client.set_timeout(5.0)
        except Exception: pass
    _cleanup_note(report)
    return report
