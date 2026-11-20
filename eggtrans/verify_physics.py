"""Run actual MCX against the analytic Beer-Lambert non-scattering slab."""
import json
from .mcx_engine import backend_status,run_slab_reference,run_source_shape_reference

def main():
    ok,status=backend_status()
    if not ok:
        raise SystemExit(status)
    checks=[]
    for mu in (0.,.03,.1):
        r=run_slab_reference(mu_a_mm1=mu,thickness_mm=12,nphotons=50_000)
        checks.append({'mu_a_mm^-1':mu,**r})
    print(json.dumps(checks,indent=2))
    if any(c['relative_error']>.045 or c['detected_packets']<47_500 for c in checks):
        raise SystemExit('FAIL: real MCX slab transmission differs from Beer-Lambert reference')
    print('PASS: all real MCX slab tests match analytical Beer-Lambert within 4.5%.')
    sources=run_source_shape_reference()
    print(json.dumps(sources,indent=2))
    if not (.85<sources['Spot']['aspect_x_over_y']<1.15 and sources['Transverse line']['aspect_x_over_y']>10.):
        raise SystemExit('FAIL: native MCX spot or line source projected shape is wrong')
    print('PASS: MCX disk gives a circular projected footprint, planar gives a transverse stripe.')

if __name__=='__main__':main()
