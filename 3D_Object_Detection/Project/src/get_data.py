from tqdm import tqdm
import requests
from pathlib import Path
import json
from src.config.logconfig import logging


import functools 
import csv

from collections import namedtuple
import SimpleITK as sitk
import numpy as np
import copy
from src.utils import getCache
from src.config.config import data_path, url


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# json_config = Path('/tmp/myapp/src/config/data_config.json')
# with open(json_config,'r') as fp:
#         data_conf = json.load(fp)
#         data_path = Path(data_conf['output_dir'])

CandidateInfoTuple = namedtuple(
          'CandidateInfoTuple',
          'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

raw_cache = getCache(data_path / 'cache_raw')


IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)    
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    coord_a = np.array(coord_xyz)
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    cri_a = ((coord_a - origin_a)@np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    
    return IrcTuple(cri_a[2], cri_a[1], cri_a[0])


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = data_path.glob('subset*/*.mhd')
    presentOnDisk_set = {p.stem for p in mhd_list}

    # read diameter and coordinates from annotations file
    diameter_dict = {}
    with open(data_path / 'annotations.csv', "r") as f:
          for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
              (annotationCenter_xyz, annotationDiameter_mm)
        )

    candidateInfo_list = []
    with open(data_path / 'candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                    else:
                        candidateDiameter_mm = annotationDiameter_mm
                        break

            candidateInfo_list.append(CandidateInfoTuple(
                  isNodule_bool,
                  candidateDiameter_mm,
                  series_uid,
                  candidateCenter_xyz,
                ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = next(data_path.glob(f'subset*/{series_uid}.mhd'))             
        ct_mhd = sitk.ReadImage(mhd_path.as_posix())
        # 3D array
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class getData:
    def __init__(self, url, output_dir):
        self.url = url
        self.file_name = self.url.split('/')[-1].split('?')[0]
        self.output_path = Path(output_dir) / self.file_name

    @classmethod
    def from_json(cls, json_config):
        with open(json_config,'r') as fp:
            data_conf = json.load(fp)
        url = data_conf['url']
        output_dir = data_conf['output_dir']
        return cls(url, output_dir)

    @classmethod
    def from_config(cls):
        return cls(url, data_path)


    def stream(self):
        # Streaming, so we can iterate over the response.
        log.info(f"Starting reading {self.file_name}")
        with requests.get(self.url, stream=True) as response:
            response.raise_for_status()
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 8192
            with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
                with open(self.output_path, 'wb') as file:
                    for data in response.iter_content(block_size):
                        if data:
                            progress_bar.update(len(data))
                            file.write(data)
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    log.error("Something went wrong")        

                    
                    
                    

if __name__ == "__main__":
    #json_config = Path('src/config') / 'data_config.json'
    #getData.from_json(json_config).stream()
    getData.from_config().stream()