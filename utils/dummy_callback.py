# This dummy callback is used for writing IMS files. It does nothing, but it is needed.


class dummy_callback:
    def RecordProgress(self, percent_complete: float, bytes_remaining: int):
        pass  # No-op
