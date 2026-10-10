CSE 168 Homework 1 - Ray Tracing

1) Chris Enotiadis

2) Website: https://bisbouras.github.io/cse168-final/


3) Built on top of my HW4 path tracer (MIS, GGX, NEE, russian roulette).
   Added homogeneous participating media: absorption, scattering, and MIS
   direct lighting inside the medium.
 
   Medium: a global homogeneous Medium on the scene with sigma_a, sigma_s,
   and HG asymmetry g. sigma_t = sigma_a + sigma_s. Parser tokens sigma_a,
   sigma_s, g. Scenes without a medium render identically to HW4.
 
   Absorption only (sigma_s = 0): beer-lambert transmittance exp(-sigma_t*d)
   multiplied into throughput on every path segment, and onto NEE / BRDF
   light samples over their distance to the light.
 
   Free flight sampling (sigma_s > 0): each bounce samples
   t = -log(1-xi) / sigma_t_s with sigma_t_s the mean of the rgb channels.
   t < hit.t means the vertex is a scattering event in the medium instead
   of the surface hit. The scatter vertex weight is
   (sigma_s / sigma_t_s) * exp((sigma_t_s - sigma_t) * t) and the
   no-scatter weight is exp((sigma_t_s - sigma_t) * t), which keeps the
   estimator unbiased for colored media despite sampling with a scalar.
 
   Henyey-Greenstein: hgPhase evaluates the closed form, sampleHG inverts
   the CDF on cosTheta (isotropic special case at g near 0). The HG value
   is also the solid angle pdf since the sampling is exact.
 
   MIS in the medium: shadeMISVolume mirrors the surface shadeMIS with the
   phase function in the BRDF role and no surface cosine. One NEE sample
   per light plus one phase sample, power heuristic with sample counts.
   NEE shadow rays pick up transmittance over their length. The phase side
   simplifies to w * Li * transmittance because p_HG / pdf = 1. Path
   continuation also samples HG, the pdf cancels against the phase
   function so the throughput only carries the scatter vertex weight.
 
   Hero scene: dark room, four pane window, small very bright quad light
   far outside so the source angular size stays small and the mullion
   shadows stay crisp inside the beam. sigma_a 0.02, sigma_s 0.075,
   g 0.45. Geometry emitted by scenes/gen_hero.py. Final render 800x800
   at 4096spp. Remaining speckle is fireflies from in-fog paths connecting
   to the small bright light, kept rather than clamping to stay unbiased.
 
   Stretch goal (heterogeneous media via delta tracking) was scoped as
   conditional in the proposal and did not happen.