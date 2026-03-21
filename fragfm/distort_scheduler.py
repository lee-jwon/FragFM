import torch


class DistortScheduler:
    def __init__(self, distort_type: str):
        self.distort_type = distort_type.lower()
        valid_distortions = [
            "polyinc",
            "cos",
            "identity",
            "revcos",
            "polydec",
            "polydec2",
            "polydec3",
            "polydec4",
        ]
        if self.distort_type not in valid_distortions:
            raise ValueError(f"distort_type must be one of {valid_distortions}.")

    def convert_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.distort_type == "polyinc":
            return t**2
        elif self.distort_type == "cos":
            return (1 - torch.cos(torch.pi * t)) / 2
        elif self.distort_type == "identity":
            return t
        elif self.distort_type == "revcos":
            return 2 * t - (1 - torch.cos(torch.pi * t)) / 2
        elif self.distort_type == "polydec" or self.distort_type == "polydec2":
            return 2 * t - (t**2)
        elif self.distort_type == "polydec3":
            return (t**3) - 3 * (t**2) + 3 * t
        elif self.distort_type == "polydec4":
            return 1 - ((1 - t) ** 4)
        else:
            raise ValueError(f"Invalid distortion type: {self.distort_type}")

    def __str__(self):
        return f"DistortScheduler(distort_type={self.distort_type})"

    def __repr__(self):
        return self.__str__()


# Example usage:
if __name__ == "__main__":
    distorter = DistortScheduler("cos")
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    distorted_t = distorter.get_distort(t)
    print("t =         ", t)
    print("distorted = ", distorted_t)
