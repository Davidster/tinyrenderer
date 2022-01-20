use super::helpers::*;

use anyhow::anyhow;

#[derive(Clone, Debug)]
pub struct Segment3D {
    pub p1: Pointf,
    pub p2: Pointf,
}

impl Segment3D {
    pub fn new(p1: Pointf, p2: Pointf) -> Self {
        Segment3D { p1, p2 }
    }
    pub fn get_point(&self, t: f64) -> anyhow::Result<Pointf> {
        if t < 0.0 || t > 1.0 {
            return Err(anyhow!(
                "t must be in the range [0.0, 1.0]. Got t = {:?}",
                t
            ));
        }
        let d_x = self.p2.x - self.p1.x;
        let d_y = self.p2.y - self.p1.y;
        let d_z = self.p2.z - self.p1.z;
        let result = Pointf {
            x: self.p1.x + t * d_x,
            y: self.p1.y + t * d_y,
            z: self.p1.z + t * d_z,
        };
        // println!("get_point: t = {:?}, result = {:?}", t, result);
        Ok(result)
    }
    pub fn get_point_iterator(
        &self,
        resolution_arg: impl Into<Option<i64>>,
    ) -> anyhow::Result<SegmentRange> {
        let resolution = resolution_arg.into().unwrap_or_else(|| {
            (((self.p2.x - self.p1.x)
                .abs()
                .max((self.p2.y - self.p1.y).abs())
                .ceil() as i64)
                + 1)
            .max(2)
        });
        // dbg!(resolution);
        if resolution < 2 {
            return Err(anyhow!(
                "resolution must be > 1. Got resolution = {:?}",
                resolution
            ));
        }
        Ok(SegmentRange {
            segment: self.clone(),
            resolution,
            i: 0,
        })
    }
}

pub struct SegmentRange {
    segment: Segment3D,
    resolution: i64,
    i: i64,
}

impl Iterator for SegmentRange {
    type Item = Pointf;

    fn next(&mut self) -> Option<Pointf> {
        if self.i > self.resolution {
            None
        } else {
            let delta = 1.0 / self.resolution as f64;
            let result = Some(self.segment.get_point(self.i as f64 * delta).unwrap());
            self.i += 1;
            result
        }
    }
}
